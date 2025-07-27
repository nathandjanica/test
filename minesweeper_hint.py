import cv2 as cv
import numpy as np
import pyautogui
import time
import keyboard
import os
import ctypes
from ctypes import wintypes
from collections import defaultdict
import re
import random
from typing import List, Tuple, Set, Dict, Optional

# === Config ===
GRID_LINE_COLOR = np.array([128, 128, 129])
GRID_TOLERANCE = 10
BORDER_COLOR = np.array([30, 30, 30])
BORDER_TOLERANCE = 10

X1, Y1, X2, Y2 = 716, 317, 1203, 804
CHECK_DELAY = 0.01  # near realtime update

TEMPLATE_DIR = "templates"
TILE_MATCH_THRESHOLD = 0.75
BASE_TEMPLATE_SIZE = 18

TARGET_RGB = (163, 163, 163)  # "?" pixel color

# Globals
DETECTION_ENABLED = True
RUNNING = True
last_click_pos: Optional[Tuple[int, int]] = None
remaining_mines: Optional[int] = None

tiles: List[Tuple[int, int, int, int]] = []
safe_tiles: Set[Tuple[int, int]] = set()
mine_tiles: Set[Tuple[int, int]] = set()
grid: List[List[str]] = []
inactive_tiles: Set[int] = set()
first_scan: bool = True

def load_templates(directory: str) -> Dict[str, List[np.ndarray]]:
    templates = defaultdict(list)
    for fname in os.listdir(directory):
        name, ext = os.path.splitext(fname)
        if ext.lower() in ['.png', '.jpg']:
            match = re.match(r'(\d+|flag)', name.lower())
            label = match.group(1) if match else name.lower()
            img = cv.imread(os.path.join(directory, fname), cv.IMREAD_GRAYSCALE)
            if img is not None:
                templates[label].append(img)
            else:
                print(f"Warning: Could not load template {fname}")
    return templates

TEMPLATES = load_templates(TEMPLATE_DIR)

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]

class MouseController:
    INPUT_MOUSE = 0
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_ABSOLUTE = 0x8000
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010

    @staticmethod
    def _get_abs_coords(x, y):
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        abs_x = int(x * 65535 / (screen_width - 1))
        abs_y = int(y * 65535 / (screen_height - 1))
        return abs_x, abs_y

    @staticmethod
    def move(x, y):
        abs_x, abs_y = MouseController._get_abs_coords(x, y)
        input_move = INPUT(
            type=MouseController.INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=abs_x,
                dy=abs_y,
                mouseData=0,
                dwFlags=MouseController.MOUSEEVENTF_MOVE | MouseController.MOUSEEVENTF_ABSOLUTE,
                time=0,
                dwExtraInfo=None
            )
        )
        ctypes.windll.user32.SendInput(1, ctypes.byref(input_move), ctypes.sizeof(INPUT))

    @staticmethod
    def click(x, y, right=False):
        abs_x, abs_y = MouseController._get_abs_coords(x, y)
        flags_down = MouseController.MOUSEEVENTF_RIGHTDOWN if right else MouseController.MOUSEEVENTF_LEFTDOWN
        flags_up = MouseController.MOUSEEVENTF_RIGHTUP if right else MouseController.MOUSEEVENTF_LEFTUP

        def send(flags):
            input_event = INPUT(
                type=MouseController.INPUT_MOUSE,
                mi=MOUSEINPUT(
                    dx=abs_x,
                    dy=abs_y,
                    mouseData=0,
                    dwFlags=MouseController.MOUSEEVENTF_MOVE | MouseController.MOUSEEVENTF_ABSOLUTE | flags,
                    time=0,
                    dwExtraInfo=None
                )
            )
            ctypes.windll.user32.SendInput(1, ctypes.byref(input_event), ctypes.sizeof(INPUT))

        send(0)
        time.sleep(0.02)
        send(flags_down)
        time.sleep(0.02)
        send(flags_up)
        time.sleep(0.02)

def jitter_pos(x, y, jitter=2):
    return x + random.randint(-jitter, jitter), y + random.randint(-jitter, jitter)

def safe_mouse_click_retry(x, y, right=False, max_retries=3):
    global last_click_pos
    for attempt in range(max_retries):
        if last_click_pos is None or (abs(last_click_pos[0] - x) > 1 or abs(last_click_pos[1] - y) > 1):
            MouseController.click(x, y, right)
            last_click_pos = (x, y)
            time.sleep(0.05)
            return True
        else:
            jittered_x, jittered_y = jitter_pos(x, y)
            MouseController.move(jittered_x, jittered_y)
            time.sleep(0.03)
    return False

def auto_flag_tiles(tiles, mine_tiles, grid_width, grid, gray, cropped_bgr):
    MouseController.move(794, 296)
    time.sleep(0.1)
    # Only attempt to flag one tile per call for incremental re-evaluation
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
        x, y, w, h = tiles[row * grid_width + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        jittered_x, jittered_y = jitter_pos(click_x, click_y)
        safe_mouse_click_retry(jittered_x, jittered_y, right=True)
        time.sleep(0.1)  # Let game UI update

        # Re-classify this tile and its neighbors
        affected = [(row, col)] + get_neighbors(row, col, grid)
        xs = sorted(set(x for (x, y, w, h) in tiles))
        ys = sorted(set(y for (x, y, w, h) in tiles))
        x_map = {x: i for i, x in enumerate(xs)}
        y_map = {y: i for i, y in enumerate(ys)}
        for nr, nc in affected:
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                idx = nr * grid_width + nc
                if idx < len(tiles):
                    tx, ty, tw, th = tiles[idx]
                    tile_gray = gray[ty:ty + th, tx:tx + tw]
                    tile_bgr = cropped_bgr[ty:ty + th, tx:tx + tw]
                    contains_target = np.any(np.all(tile_bgr == TARGET_RGB[::-1], axis=2))
                    label, _ = classify_tile(tile_gray, tw, th)
                    if label == "flag":
                        grid[nr][nc] = "F"
                    elif contains_target:
                        grid[nr][nc] = "?"
                    elif label == "unknown":
                        grid[nr][nc] = "0"
                    else:
                        grid[nr][nc] = label
        MouseController.move(794, 296)
        break  # Only one flag per call
    MouseController.move(794, 296)

def auto_click_tiles(tiles, safe_tiles, grid_width):
    MouseController.move(794, 296)
    time.sleep(0.1)
    for (row, col) in safe_tiles:
        x, y, w, h = tiles[row * grid_width + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        jittered_x, jittered_y = jitter_pos(click_x, click_y)
        safe_mouse_click_retry(jittered_x, jittered_y, right=False)
        time.sleep(0.07)
    MouseController.move(794, 296)

def find_grid_lines(image: np.ndarray) -> Tuple[List[int], List[int], np.ndarray]:
    lower_main = np.clip(GRID_LINE_COLOR - GRID_TOLERANCE, 0, 255)
    upper_main = np.clip(GRID_LINE_COLOR + GRID_TOLERANCE, 0, 255)
    lower_border = np.clip(BORDER_COLOR - BORDER_TOLERANCE, 0, 255)
    upper_border = np.clip(BORDER_COLOR + BORDER_TOLERANCE, 0, 255)

    mask_main = cv.inRange(image, lower_main, upper_main)
    mask_border = cv.inRange(image, lower_border, upper_border)
    mask = cv.bitwise_or(mask_main, mask_border)

    vertical_proj = np.sum(mask, axis=0)
    horizontal_proj = np.sum(mask, axis=1)

    vertical_lines = np.where(vertical_proj > np.max(vertical_proj) * 0.5)[0]
    horizontal_lines = np.where(horizontal_proj > np.max(horizontal_proj) * 0.5)[0]

    def cluster(lines, max_gap=5):
        if not len(lines):
            return []
        grouped = []
        group = [lines[0]]
        for x in lines[1:]:
            if x - group[-1] <= max_gap:
                group.append(x)
            else:
                grouped.append(int(np.mean(group)))
                group = [x]
        grouped.append(int(np.mean(group)))
        return grouped

    return cluster(vertical_lines), cluster(horizontal_lines), mask

def generate_tile_boxes(v_lines: List[int], h_lines: List[int]) -> List[Tuple[int, int, int, int]]:
    tiles = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            x, y = v_lines[j] + 1, h_lines[i] + 1
            w, h = v_lines[j + 1] - v_lines[j] - 1, h_lines[i + 1] - h_lines[i] - 1
            if 10 < w < 100 and 10 < h < 100:
                tiles.append((x, y, w, h))
    return tiles

def classify_tile(tile_img_gray, tile_w, tile_h):
    best_score = 0
    best_label = "unknown"
    base_size = BASE_TEMPLATE_SIZE
    avg_scale = (base_size / tile_w + base_size / tile_h) / 2
    scales = [avg_scale * s for s in [0.85, 0.95, 1.0, 1.05, 1.15]]

    for label, templates in TEMPLATES.items():
        for tmpl in templates:
            for scale in scales:
                try:
                    resized = cv.resize(tile_img_gray, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
                except cv.error:
                    continue
                if resized.shape[0] < tmpl.shape[0] or resized.shape[1] < tmpl.shape[1]:
                    continue
                result = cv.matchTemplate(resized, tmpl, cv.TM_CCOEFF_NORMED)
                _, score, _, _ = cv.minMaxLoc(result)
                if score > best_score:
                    best_score = score
                    best_label = label

    return (best_label if best_score >= TILE_MATCH_THRESHOLD else "unknown"), best_score

def build_grid(tiles, labels):
    if not tiles or not labels:
        return []

    xs = sorted(set(x for (x, y, w, h) in tiles))
    ys = sorted(set(y for (x, y, w, h) in tiles))
    if not xs or not ys:
        return []

    x_map = {x: i for i, x in enumerate(xs)}
    y_map = {y: i for i, y in enumerate(ys)}
    width, height = len(xs), len(ys)

    grid = [["0" for _ in range(width)] for _ in range(height)]
    for (x, y, w, h), label in zip(tiles, labels):
        row, col = y_map[y], x_map[x]
        if label in ("unknown", "u"):
            label = "0"
        elif label == "?":
            label = "?"
        elif label in ("flag", "f"):
            label = "F"
        grid[row][col] = label

    return grid

def get_neighbors(row, col, grid):
    rows, cols = len(grid), len(grid[0])
    return [
        (r, c)
        for dr in [-1, 0, 1]
        for dc in [-1, 0, 1]
        if not (dr == 0 and dc == 0)
        if 0 <= (r := row + dr) < rows and 0 <= (c := col + dc) < cols
    ]

def is_tile_active(row, col, grid):
    """Returns True if tile is unknown and has a neighbor that is a number or flagged."""
    if grid[row][col] != "?":
        return False
    for nr, nc in get_neighbors(row, col, grid):
        if grid[nr][nc] in "12345678F":
            return True
    return False

def count_neighbors(row, col, grid, value):
    return sum(1 for r, c in get_neighbors(row, col, grid) if grid[r][c] == value)

def get_unknown_neighbors(row, col, grid):
    return [(r, c) for r, c in get_neighbors(row, col, grid) if grid[r][c] == "?"]

def solver_step(grid):
    safe, mines = set(), set()
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val in "12345678":
                n = int(val)
                flagged = count_neighbors(r, c, grid, "F")
                unknown = count_neighbors(r, c, grid, "?")
                unknown_neighbors = get_unknown_neighbors(r, c, grid)

                if n == flagged and unknown > 0:
                    safe.update(unknown_neighbors)
                if n - flagged == unknown and unknown > 0:
                    mines.update(unknown_neighbors)
    return safe, mines

def compute_probabilities(grid, remaining_mines):
    rows, cols = len(grid), len(grid[0])
    prob_grid = [[0.0 for _ in range(cols)] for _ in range(rows)]
    total_unknown = sum(row.count("?") for row in grid)
    if total_unknown == 0:
        return prob_grid
    uniform_prob = remaining_mines / total_unknown
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "?":
                prob_grid[r][c] = uniform_prob
    return prob_grid

def set_detection(state: bool):
    global DETECTION_ENABLED
    DETECTION_ENABLED = state
    print("▶️  Detection resumed" if state else "⏸️  Detection paused")

def quit_program():
    global RUNNING
    print("❌ Quitting")
    RUNNING = False

def auto_flag_action():
    global tiles, mine_tiles, grid, gray, cropped_bgr
    if not DETECTION_ENABLED:
        print("Detection paused. Cannot auto-flag.")
        return
    if not tiles or not mine_tiles or not grid:
        print("No mine tiles to flag.")
        return
    print("🚩 Auto-flagging one mine, then rechecking just that tile and its neighbors...")
    auto_flag_tiles(tiles, mine_tiles, len(grid[0]), grid, gray, cropped_bgr)

def auto_click_action():
    global tiles, safe_tiles, grid
    if not DETECTION_ENABLED:
        print("Detection paused. Cannot auto-click.")
        return
    if not tiles or not safe_tiles or not grid:
        print("No safe tiles to click.")
        return
    print("🟩 Auto-clicking safe tiles...")
    auto_click_tiles(tiles, safe_tiles, len(grid[0]))

def prompt_remaining_mines():
    global remaining_mines
    try:
        rem_str = input("Enter remaining mines (integer): ")
        remaining_mines = int(rem_str)
        print(f"Remaining mines set to {remaining_mines}")
    except Exception as e:
        print(f"Invalid input for remaining mines: {e}")
        remaining_mines = None

keyboard.add_hotkey('alt+p', lambda: set_detection(False), suppress=True)
keyboard.add_hotkey('alt+r', lambda: set_detection(True), suppress=True)
keyboard.add_hotkey('alt+q', quit_program, suppress=True)
keyboard.add_hotkey('alt+f', auto_flag_action, suppress=True)
keyboard.add_hotkey('alt+k', auto_click_action, suppress=True)
keyboard.add_hotkey('ctrl+p', prompt_remaining_mines, suppress=True)

def main():
    global DETECTION_ENABLED, RUNNING, tiles, safe_tiles, mine_tiles, grid, remaining_mines, inactive_tiles, first_scan, gray, cropped_bgr

    width, height = X2 - X1, Y2 - Y1

    cv.namedWindow("Detected Tiles + Solver Hints", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Detected Tiles + Solver Hints", cv.WND_PROP_TOPMOST, 1)
    cv.resizeWindow("Detected Tiles + Solver Hints", width, height)

    prev_labels = []
    first_scan = True
    while RUNNING:
        start_time = time.time()

        screenshot = np.array(pyautogui.screenshot())
        cropped = screenshot[Y1:Y2, X1:X2]
        cropped_bgr = cv.cvtColor(cropped, cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(cropped_bgr, cv.COLOR_BGR2GRAY)

        if DETECTION_ENABLED:
            v_lines, h_lines, _ = find_grid_lines(cropped_bgr)
            if len(v_lines) < 2 or len(h_lines) < 2:
                safe_tiles.clear(); mine_tiles.clear()
                tiles.clear()
                grid.clear()
                cv.imshow("Detected Tiles + Solver Hints", cropped_bgr)
                time.sleep(CHECK_DELAY)
                continue

            tiles = generate_tile_boxes(v_lines, h_lines)
            # Always classify all tiles on first scan
            labels = []
            if first_scan or not prev_labels or not grid or not grid[0]:
                for (x, y, w, h) in tiles:
                    tile_gray = gray[y:y + h, x:x + w]
                    tile_bgr = cropped_bgr[y:y + h, x:x + w]
                    contains_target = np.any(np.all(tile_bgr == TARGET_RGB[::-1], axis=2))
                    label, _ = classify_tile(tile_gray, w, h)
                    if label == "flag":
                        display_label = "F"
                    elif contains_target:
                        display_label = "?"
                    elif label == "unknown":
                        display_label = "0"
                    else:
                        display_label = label
                    labels.append(display_label)
                first_scan = False
            else:
                temp_grid = build_grid(tiles, prev_labels)
                xs = sorted(set(x for (x, y, w, h) in tiles))
                ys = sorted(set(y for (x, y, w, h) in tiles))
                x_map = {x: i for i, x in enumerate(xs)}
                y_map = {y: i for i, y in enumerate(ys)}
                labels = []
                inactive_tiles.clear()
                for idx, (x, y, w, h) in enumerate(tiles):
                    row = y_map[y]
                    col = x_map[x]
                    if is_tile_active(row, col, temp_grid):
                        tile_gray = gray[y:y + h, x:x + w]
                        tile_bgr = cropped_bgr[y:y + h, x:x + w]
                        contains_target = np.any(np.all(tile_bgr == TARGET_RGB[::-1], axis=2))
                        label, _ = classify_tile(tile_gray, w, h)
                        if label == "flag":
                            display_label = "F"
                        elif contains_target:
                            display_label = "?"
                        elif label == "unknown":
                            display_label = "0"
                        else:
                            display_label = label
                        labels.append(display_label)
                    else:
                        # Use previous label if present, else assign "cached"
                        if idx < len(prev_labels):
                            labels.append(prev_labels[idx])
                        else:
                            labels.append("cached")
                        inactive_tiles.add(idx)
            prev_labels = labels.copy()

            grid = build_grid(tiles, labels)
            if not grid or not grid[0]:
                safe_tiles.clear(); mine_tiles.clear()
            else:
                safe_tiles, mine_tiles = solver_step(grid)

            debug_img = cropped_bgr.copy()
            xs = sorted(set(x for (x, y, w, h) in tiles))
            ys = sorted(set(y for (x, y, w, h) in tiles))
            x_map = {x: i for i, x in enumerate(xs)}
            y_map = {y: i for i, y in enumerate(ys)}
            for idx, (x, y, w, h) in enumerate(tiles):
                row = y_map[y]
                col = x_map[x]
                if idx in inactive_tiles:
                    # Draw orange for inactive/skipped tiles
                    cv.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                else:
                    label = labels[idx]
                    cv.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv.putText(debug_img, label, (x + 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            for (r, c) in safe_tiles:
                if r * len(grid[0]) + c < len(tiles):
                    x, y, w, h = tiles[r * len(grid[0]) + c]
                    cv.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for (r, c) in mine_tiles:
                if r * len(grid[0]) + c < len(tiles):
                    x, y, w, h = tiles[r * len(grid[0]) + c]
                    cv.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(debug_img, "M", (x + 2, y + h - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Probability hints if no forced moves
            if remaining_mines is not None and not safe_tiles and not mine_tiles:
                prob_grid = compute_probabilities(grid, remaining_mines)
                unknown_probs = [((r, c), prob_grid[r][c])
                                 for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == "?"]
                for (r, c), p in sorted(unknown_probs, key=lambda x: x[1])[:3]:
                    if r * len(grid[0]) + c < len(tiles):
                        x, y, w, h = tiles[r * len(grid[0]) + c]
                        cv.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv.putText(debug_img, f"{int(p*100)}%", (x + 2, y + h - 4),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv.imshow("Detected Tiles + Solver Hints", debug_img)

        else:
            time.sleep(CHECK_DELAY)

        if cv.waitKey(1) & 0xFF == 27:
            break

        elapsed = time.time() - start_time
        if elapsed < CHECK_DELAY:
            time.sleep(CHECK_DELAY - elapsed)

    cv.destroyAllWindows()

if __name__ == "__main__":
    print("Hotkeys: Alt+P=Pause, Alt+R=Resume, Alt+Q=Quit, Alt+F=Auto-Flag, Alt+K=Auto-Click Safe, Ctrl+P=Set Remaining Mines")
    main()
