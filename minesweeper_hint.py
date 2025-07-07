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

# === Config ===
GRID_LINE_COLOR = np.array([128, 128, 129])
GRID_TOLERANCE = 10
BORDER_COLOR = np.array([30, 30, 30])
BORDER_TOLERANCE = 10

X1, Y1, X2, Y2 = 716, 317, 1203, 804
CHECK_DELAY = 0.01  # near realtime update
DETECTION_ENABLED = True
RUNNING = True

TEMPLATE_DIR = "templates"
TILE_MATCH_THRESHOLD = 0.75
BASE_TEMPLATE_SIZE = 18

last_click_pos = None
remaining_mines = None  # set by user input on demand

TARGET_RGB = (163, 163, 163)  # "?" pixel color

# Load multiple templates per label (numbers only)
TEMPLATES = defaultdict(list)
for fname in os.listdir(TEMPLATE_DIR):
    name, ext = os.path.splitext(fname)
    if ext.lower() in ['.png', '.jpg']:
        match = re.match(r'(\d+|flag)', name.lower())
        label = match.group(1) if match else name.lower()
        img = cv.imread(os.path.join(TEMPLATE_DIR, fname), cv.IMREAD_GRAYSCALE)
        if img is not None:
            TEMPLATES[label].append(img)
        else:
            print(f"Warning: Could not load template {fname}")

# === Windows ctypes mouse input setup ===
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

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

SendInput = ctypes.windll.user32.SendInput

def _get_abs_coords(x, y):
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    abs_x = int(x * 65535 / (screen_width - 1))
    abs_y = int(y * 65535 / (screen_height - 1))
    return abs_x, abs_y

def mouse_move(x, y):
    abs_x, abs_y = _get_abs_coords(x, y)
    input_move = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(
        dx=abs_x,
        dy=abs_y,
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
        time=0,
        dwExtraInfo=None
    ))
    SendInput(1, ctypes.byref(input_move), ctypes.sizeof(INPUT))

def mouse_click(x, y, right=False):
    abs_x, abs_y = _get_abs_coords(x, y)

    input_move = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(
        dx=abs_x,
        dy=abs_y,
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
        time=0,
        dwExtraInfo=None
    ))

    flags_down = MOUSEEVENTF_RIGHTDOWN if right else MOUSEEVENTF_LEFTDOWN
    input_down = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(
        dx=abs_x,
        dy=abs_y,
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | flags_down,
        time=0,
        dwExtraInfo=None
    ))

    flags_up = MOUSEEVENTF_RIGHTUP if right else MOUSEEVENTF_LEFTUP
    input_up = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(
        dx=abs_x,
        dy=abs_y,
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | flags_up,
        time=0,
        dwExtraInfo=None
    ))

    SendInput(1, ctypes.byref(input_move), ctypes.sizeof(INPUT))
    time.sleep(0.02)
    SendInput(1, ctypes.byref(input_down), ctypes.sizeof(INPUT))
    time.sleep(0.02)
    SendInput(1, ctypes.byref(input_up), ctypes.sizeof(INPUT))
    time.sleep(0.02)

def jitter_pos(x, y, jitter=2):
    return x + random.randint(-jitter, jitter), y + random.randint(-jitter, jitter)

def safe_mouse_click_retry(x, y, right=False, max_retries=3):
    global last_click_pos

    for attempt in range(max_retries):
        if last_click_pos is None or (abs(last_click_pos[0] - x) > 1 or abs(last_click_pos[1] - y) > 1):
            mouse_click(x, y, right)
            last_click_pos = (x, y)
            time.sleep(0.05)
            return True
        else:
            jittered_x, jittered_y = jitter_pos(x, y)
            mouse_move(jittered_x, jittered_y)
            time.sleep(0.03)
    return False

def auto_flag_mines(tiles, mine_tiles, grid_width):
    mouse_move(794, 296)
    time.sleep(0.1)
    for (r, c) in mine_tiles:
        x, y, w, h = tiles[r * grid_width + c]
        click_x = X1 + x + w // 2
        click_y = Y1 + y + h // 2
        jittered_x, jittered_y = jitter_pos(click_x, click_y)
        safe_mouse_click_retry(jittered_x, jittered_y, right=True)
        time.sleep(0.07)
    mouse_move(794, 296)

def auto_click_safe(tiles, safe_tiles, grid_width):
    mouse_move(794, 296)
    time.sleep(0.1)
    for (r, c) in safe_tiles:
        x, y, w, h = tiles[r * grid_width + c]
        click_x = X1 + x + w // 2
        click_y = Y1 + y + h // 2
        jittered_x, jittered_y = jitter_pos(click_x, click_y)
        safe_mouse_click_retry(jittered_x, jittered_y, right=False)
        time.sleep(0.07)
    mouse_move(794, 296)

def find_grid_lines_positions(image):
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

    def cluster_lines(lines, max_gap=5):
        if len(lines) == 0:
            return []
        clustered = []
        group = [lines[0]]
        for x in lines[1:]:
            if x - group[-1] <= max_gap:
                group.append(x)
            else:
                clustered.append(int(np.mean(group)))
                group = [x]
        clustered.append(int(np.mean(group)))
        return clustered

    v_lines = cluster_lines(vertical_lines)
    h_lines = cluster_lines(horizontal_lines)
    return v_lines, h_lines, mask

def generate_tile_coordinates(v_lines, h_lines):
    tiles = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            x = v_lines[j] + 1
            y = h_lines[i] + 1
            w = v_lines[j + 1] - v_lines[j] - 1
            h = h_lines[i + 1] - h_lines[i] - 1
            if 10 < w < 100 and 10 < h < 100:
                tiles.append((x, y, w, h))
    return tiles

def classify_tile(tile_img_gray, tile_w, tile_h):
    best_score = 0
    best_label = "unknown"
    base_size = BASE_TEMPLATE_SIZE

    scale_factor_w = base_size / tile_w
    scale_factor_h = base_size / tile_h
    avg_scale_factor = (scale_factor_w + scale_factor_h) / 2

    scales = [avg_scale_factor * s for s in [0.85, 0.95, 1.0, 1.05, 1.15]]

    for label, tmpl_list in TEMPLATES.items():
        for tmpl in tmpl_list:
            for scale in scales:
                try:
                    resized_tile = cv.resize(tile_img_gray, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
                except cv.error:
                    continue
                if resized_tile.shape[0] < tmpl.shape[0] or resized_tile.shape[1] < tmpl.shape[1]:
                    continue
                result = cv.matchTemplate(resized_tile, tmpl, cv.TM_CCOEFF_NORMED)
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

    width = len(xs)
    height = len(ys)

    grid = [["0" for _ in range(width)] for __ in range(height)]

    for (x, y, w, h), label in zip(tiles, labels):
        row = y_map[y]
        col = x_map[x]
        if label == "unknown" or label == "u":
            label = "0"
        elif label == "?":
            label = "?"
        elif label == "flag" or label == "f":
            label = "F"
        else:
            label = label  # number as string
        grid[row][col] = label

    return grid

def get_neighbors(r, c, grid):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbors.append((nr, nc))
    return neighbors

def count_flagged_neighbors(r, c, grid):
    neighbors = get_neighbors(r, c, grid)
    return sum(1 for nr, nc in neighbors if grid[nr][nc] == "F")

def count_unknown_neighbors(r, c, grid):
    neighbors = get_neighbors(r, c, grid)
    return sum(1 for nr, nc in neighbors if grid[nr][nc] == "?")

def get_unknown_neighbors(r, c, grid):
    neighbors = get_neighbors(r, c, grid)
    return [(nr, nc) for nr, nc in neighbors if grid[nr][nc] == "?"]

def solver_step(grid):
    safe_tiles = set()
    mine_tiles = set()
    height = len(grid)
    width = len(grid[0])

    for r in range(height):
        for c in range(width):
            val = grid[r][c]
            if val in "12345678":
                number = int(val)
                flagged = count_flagged_neighbors(r, c, grid)
                unknown = count_unknown_neighbors(r, c, grid)
                unknown_neighbors = get_unknown_neighbors(r, c, grid)

                if number == flagged and unknown > 0:
                    safe_tiles.update(unknown_neighbors)

                if number - flagged == unknown and unknown > 0:
                    mine_tiles.update(unknown_neighbors)

    return safe_tiles, mine_tiles

def compute_probabilities(grid, remaining_mines):
    height = len(grid)
    width = len(grid[0])
    prob_grid = [[0.0 for _ in range(width)] for __ in range(height)]

    total_unknown = sum(row.count("?") for row in grid)
    if total_unknown == 0:
        return prob_grid

    uniform_prob = remaining_mines / total_unknown
    for r in range(height):
        for c in range(width):
            if grid[r][c] == "?":
                prob_grid[r][c] = uniform_prob
    return prob_grid

# Hotkeys
def on_pause():
    global DETECTION_ENABLED
    DETECTION_ENABLED = False
    print("⏸️  Detection paused")

def on_resume():
    global DETECTION_ENABLED
    DETECTION_ENABLED = True
    print("▶️  Detection resumed")

def on_quit():
    global RUNNING
    print("❌ Quitting")
    RUNNING = False

def on_auto_flag():
    global tiles, mine_tiles, grid
    if not DETECTION_ENABLED:
        print("Detection paused. Cannot auto-flag.")
        return
    if not tiles or not mine_tiles or not grid:
        print("No mine tiles to flag.")
        return
    print("🚩 Auto-flagging mines...")
    auto_flag_mines(tiles, mine_tiles, len(grid[0]))

def on_auto_safe_click():
    global tiles, safe_tiles, grid
    if not DETECTION_ENABLED:
        print("Detection paused. Cannot auto-click.")
        return
    if not tiles or not safe_tiles or not grid:
        print("No safe tiles to click.")
        return
    print("🟩 Auto-clicking safe tiles...")
    auto_click_safe(tiles, safe_tiles, len(grid[0]))

def prompt_remaining_mines():
    global remaining_mines
    try:
        rem_str = input("Enter remaining mines (integer): ")
        remaining_mines = int(rem_str)
        print(f"Remaining mines set to {remaining_mines}")
    except Exception as e:
        print(f"Invalid input for remaining mines: {e}")
        remaining_mines = None

keyboard.add_hotkey('alt+p', on_pause, suppress=True, trigger_on_release=False)
keyboard.add_hotkey('alt+r', on_resume, suppress=True, trigger_on_release=False)
keyboard.add_hotkey('alt+q', on_quit, suppress=True, trigger_on_release=False)
keyboard.add_hotkey('alt+f', on_auto_flag, suppress=True, trigger_on_release=False)
keyboard.add_hotkey('alt+k', on_auto_safe_click, suppress=True, trigger_on_release=False)
keyboard.add_hotkey('ctrl+p', prompt_remaining_mines, suppress=True, trigger_on_release=False)

tiles = []
safe_tiles = set()
mine_tiles = set()
grid = []

def main():
    global DETECTION_ENABLED, RUNNING, tiles, safe_tiles, mine_tiles, grid, remaining_mines

    width = X2 - X1
    height = Y2 - Y1

    cv.namedWindow("Detected Tiles + Solver Hints", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Detected Tiles + Solver Hints", cv.WND_PROP_TOPMOST, 1)
    cv.resizeWindow("Detected Tiles + Solver Hints", width, height)

    while RUNNING:
        start_time = time.time()

        screenshot = np.array(pyautogui.screenshot())
        cropped = screenshot[Y1:Y2, X1:X2]
        cropped_bgr = cv.cvtColor(cropped, cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(cropped_bgr, cv.COLOR_BGR2GRAY)

        if DETECTION_ENABLED:
            v_lines, h_lines, mask = find_grid_lines_positions(cropped_bgr)
            if len(v_lines) < 2 or len(h_lines) < 2:
                safe_tiles, mine_tiles = set(), set()
                tiles = []
                grid = []
                debug_img = cropped_bgr.copy()
                cv.imshow("Detected Tiles + Solver Hints", debug_img)
                time.sleep(CHECK_DELAY)
                continue

            tiles = generate_tile_coordinates(v_lines, h_lines)

            labels = []
            for (x, y, w, h) in tiles:
                tile_gray = gray[y:y + h, x:x + w]
                tile_bgr = cropped_bgr[y:y + h, x:x + w]
                contains_target = np.any(np.all(tile_bgr == TARGET_RGB[::-1], axis=2))

                label, score = classify_tile(tile_gray, w, h)
                if label == "flag":
                    display_label = "F"
                elif contains_target:
                    display_label = "?"
                elif label == "unknown":
                    display_label = "0"
                else:
                    display_label = label

                labels.append(display_label)

            grid = build_grid(tiles, labels)
            if not grid or not grid[0]:
                safe_tiles, mine_tiles = set(), set()
            else:
                safe_tiles, mine_tiles = solver_step(grid)

            debug_img = cropped_bgr.copy()
            for idx, (x, y, w, h) in enumerate(tiles):
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

            if remaining_mines is not None and len(safe_tiles) == 0 and len(mine_tiles) == 0:
                prob_grid = compute_probabilities(grid, remaining_mines)
                sorted_probs = []
                for r in range(len(grid)):
                    for c in range(len(grid[0])):
                        if grid[r][c] == "?":
                            sorted_probs.append(((r, c), prob_grid[r][c]))
                sorted_probs.sort(key=lambda x: x[1])
                top3 = sorted_probs[:3]

                for (r, c), p in top3:
                    if r * len(grid[0]) + c < len(tiles):
                        x, y, w, h = tiles[r * len(grid[0]) + c]
                        cv.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv.putText(debug_img, f"{int(p*100)}%", (x + 2, y + h - 4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
