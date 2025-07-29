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
import win32api
import win32con
import win32gui
from pynput.mouse import Button, Controller as MouseController
import threading
from queue import Queue
import itertools
import tkinter as tk  # For clipboard copy
import webbrowser  # For opening browser
import urllib.parse  # For URL encoding
import tempfile
from tkinter import simpledialog


# Check if mss is available
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("Warning: mss library not available, falling back to pyautogui for screen capture")

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
prev_labels: List[str] = []  # Add this line

## Features of Full Auto
FULL_AUTO_MODE = False

# 1. Cache pattern database at startup
PATTERN_DATABASE = None
DEBUG_MODE = False
ACTION_QUEUE = Queue()
MOUSE_POSITION_CACHE = None
MOUSE_POSITION_LAST_UPDATE = 0
TILE_CLASSIFICATION_CACHE = {}

# Debug print function
def debug_print(*args):
    if DEBUG_MODE:
        print(*args)

# Optimized mouse position with caching
def get_mouse_position():
    global MOUSE_POSITION_CACHE, MOUSE_POSITION_LAST_UPDATE
    current_time = time.time()
    if (MOUSE_POSITION_CACHE is None or 
        current_time - MOUSE_POSITION_LAST_UPDATE > 0.1):
        MOUSE_POSITION_CACHE = win32api.GetCursorPos()
        MOUSE_POSITION_LAST_UPDATE = current_time
    return MOUSE_POSITION_CACHE

# Initialize pattern database at startup
def initialize_pattern_database():
    global PATTERN_DATABASE
    if PATTERN_DATABASE is None:
        try:
            from patterns import comprehensive_solver
            PATTERN_DATABASE = {"solver": comprehensive_solver}
        except ImportError:
            PATTERN_DATABASE = {}  # Fallback if patterns module not available
    return PATTERN_DATABASE

# Optimized screen capture
def capture_screen_optimized():
    """Use more efficient screen capture with fallback"""
    if MSS_AVAILABLE:
        try:
            with mss.mss() as sct:
                monitor = {"top": Y1, "left": X1, "width": X2-X1, "height": Y2-Y1}
                screenshot = np.array(sct.grab(monitor))
                return cv.cvtColor(screenshot, cv.COLOR_BGRA2BGR)
        except Exception as e:
            debug_print(f"mss capture failed: {e}, falling back to pyautogui")
    
    # Fallback to pyautogui
    screenshot = np.array(pyautogui.screenshot())
    return cv.cvtColor(screenshot[Y1:Y2, X1:X2], cv.COLOR_RGB2BGR)

# Cached tile classification
def classify_tile_cached(tile_img_gray, tile_w, tile_h, cache_key):
    if cache_key in TILE_CLASSIFICATION_CACHE:
        return TILE_CLASSIFICATION_CACHE[cache_key]
    
    result = classify_tile(tile_img_gray, tile_w, tile_h)
    TILE_CLASSIFICATION_CACHE[cache_key] = result
    
    # Limit cache size to prevent memory leaks
    if len(TILE_CLASSIFICATION_CACHE) > 1000:
        # Clear oldest entries
        keys_to_remove = list(TILE_CLASSIFICATION_CACHE.keys())[:100]
        for key in keys_to_remove:
            del TILE_CLASSIFICATION_CACHE[key]
    
    return result

# Action queue system
def queue_action(action_type):
    """Queue actions instead of executing immediately"""
    ACTION_QUEUE.put(action_type)

def process_action_queue():
    """Process queued actions with different speed options"""
    while not ACTION_QUEUE.empty():
        action = ACTION_QUEUE.get()
        if action == 'flag':
            auto_flag_optimized()
        elif action == 'click':
            auto_click_optimized()
        elif action == 'flag_ultra':
            auto_flag_ultra_fast()
        elif action == 'click_ultra':
            auto_click_ultra_fast()
        elif action == 'flag_flow':
            auto_flag_optimized()  # Use optimized version for flow
        elif action == 'click_flow':
            auto_click_optimized()  # Use optimized version for flow

def toggle_full_auto():
    """Toggle full auto mode on/off"""
    global FULL_AUTO_MODE
    FULL_AUTO_MODE = not FULL_AUTO_MODE
    if FULL_AUTO_MODE:
        print("🤖 Full auto mode ENABLED - will continuously flag and click")
    else:
        print("⏹️ Full auto mode DISABLED")

def full_auto_cycle():
    """Ultra-fast full auto cycle - flag mines then click safe tiles"""
    global tiles, mine_tiles, safe_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not grid:
        return
    
    # First, flag any mines
    if mine_tiles:
        auto_flag_optimized()
        time.sleep(0.1)  # Minimal wait for detection to update
    
    # Then, click any safe tiles
    if safe_tiles:
        auto_click_optimized()
        time.sleep(0.1)  # Minimal wait for detection to update

def run_full_auto():
    """Ultra-fast full auto loop"""
    global FULL_AUTO_MODE, RUNNING
    
    while FULL_AUTO_MODE and RUNNING:
        try:
            full_auto_cycle()
            time.sleep(0.05)  # Ultra-fast cycle
        except KeyboardInterrupt:
            break
        except Exception as e:
            time.sleep(0.1)  # Brief error recovery

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
            # Ensure we're exactly at the target position
            MouseController.move(int(x), int(y))
            time.sleep(0.01)  # Brief pause for precision
            MouseController.click(int(x), int(y), right)
            last_click_pos = (x, y)
            time.sleep(0.05)
            return True
        else:
            jittered_x, jittered_y = jitter_pos(x, y)
            MouseController.move(jittered_x, jittered_y)
            time.sleep(0.03)
    return False

def precise_click(x, y, right=False):
    """High-precision click with position verification"""
    # Move to exact position
    MouseController.move(int(x), int(y))
    time.sleep(0.02)  # Brief pause for stability
    
    # Verify position and click
    current_x, current_y = get_mouse_position()
    if abs(current_x - x) <= 2 and abs(current_y - y) <= 2:  # Within 2 pixels
        MouseController.click(int(x), int(y), right)
        time.sleep(0.03)  # Brief pause after click
        return True
    else:
        # Retry with direct movement
        MouseController.move(int(x), int(y))
        time.sleep(0.01)
        MouseController.click(int(x), int(y), right)
        time.sleep(0.03)
        return True

def click_at_position_win32(x, y, right_click=False):
    """Click using win32api - more compatible with Roblox"""
    try:
        # Get the window handle for Roblox
        hwnd = win32gui.FindWindow(None, "Roblox")
        if hwnd == 0:
            # Try alternative window names
            hwnd = win32gui.FindWindow(None, "Minesweeper")
            if hwnd == 0:
                print("Could not find Roblox/Minesweeper window")
                return False
        
        # Bring window to front
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.1)
        
        # Convert screen coordinates to client coordinates
        client_x, client_y = win32gui.ScreenToClient(hwnd, (x, y))
        
        # Send click message
        if right_click:
            win32api.PostMessage(hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, win32api.MAKELONG(client_x, client_y))
            time.sleep(0.05)
            win32api.PostMessage(hwnd, win32con.WM_RBUTTONUP, 0, win32api.MAKELONG(client_x, client_y))
        else:
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, win32api.MAKELONG(client_x, client_y))
            time.sleep(0.05)
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, win32api.MAKELONG(client_x, client_y))
        
        return True
    except Exception as e:
        print(f"Win32 click failed: {e}")
        return False

def click_at_position_pynput(x, y, right_click=False):
    """Click using pynput with proper window focus for Roblox"""
    try:
        # Get the window handle for Roblox
        hwnd = win32gui.FindWindow(None, "Roblox")
        if hwnd == 0:
            hwnd = win32gui.FindWindow(None, "Minesweeper")
            if hwnd == 0:
                print("Could not find Roblox/Minesweeper window")
                return False
        
        # Bring window to front and activate it
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.2)  # Give time for window to activate
        
        # Use pynput for the actual clicking
        mouse = MouseController()
        mouse.position = (x, y)
        time.sleep(0.1)
        
        if right_click:
            mouse.click(Button.right, 1)
        else:
            mouse.click(Button.left, 1)
        
        return True
    except Exception as e:
        print(f"Pynput click failed: {e}")
        return False

def auto_flag_ctypes():
    """Auto-flag using ctypes for maximum compatibility"""
    global tiles, mine_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not mine_tiles or not grid:
        print("Cannot auto-flag: detection disabled or no mine tiles found")
        return
    
    print("🚩 Auto-flagging mines using ctypes...")
    
    # Move to safe position first
    MouseController.move(794, 296)
    time.sleep(0.1)
    
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Flagging mine at ({click_x}, {click_y})")
        
        try:
            # Move to position
            MouseController.move(click_x, click_y)
            time.sleep(0.05)
            
            # Right click
            MouseController.click(click_x, click_y, right=True)
            time.sleep(0.2)  # Wait for game to process
            
            print(f"  ✓ Flagged successfully")
            break  # Only flag one per call
            
        except Exception as e:
            print(f"  ✗ Failed to flag: {e}")
    
    # Move back to safe position
    MouseController.move(794, 296)
    print("Auto-flag complete")

def auto_click_ctypes():
    """Auto-click safe tiles using ctypes"""
    global tiles, safe_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not safe_tiles or not grid:
        print("Cannot auto-click: detection disabled or no safe tiles found")
        return
    
    print("🟩 Auto-clicking safe tiles using ctypes...")
    
    # Move to safe position first
    MouseController.move(794, 296)
    time.sleep(0.1)
    
    for (row, col) in safe_tiles:
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Clicking safe tile at ({click_x}, {click_y})")
        
        try:
            # Move to position
            MouseController.move(click_x, click_y)
            time.sleep(0.05)
            
            # Left click
            MouseController.click(click_x, click_y, right=False)
            time.sleep(0.15)  # Wait for game to process
            
            print(f"  ✓ Clicked successfully")
            
        except Exception as e:
            print(f"  ✗ Failed to click: {e}")
    
    # Move back to safe position
    MouseController.move(794, 296)
    print("Auto-click complete")

def auto_click_tiles_win32(tiles, safe_tiles, grid_width):
    """Auto-click using win32api for Roblox compatibility"""
    print("🟩 Auto-clicking safe tiles using win32api...")
    
    for (row, col) in safe_tiles:
        if row * grid_width + col >= len(tiles):
            continue
            
        x, y, w, h = tiles[row * grid_width + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Clicking safe tile at ({click_x}, {click_y})")
        
        if click_at_position_win32(click_x, click_y, right_click=False):
            print(f"  ✓ Click successful")
        else:
            print(f"  ✗ Click failed")
        
        time.sleep(0.15)  # Longer delay for Roblox

def auto_flag_tiles_win32(tiles, mine_tiles, grid_width, grid, gray, cropped_bgr):
    """Auto-flag using win32api for Roblox compatibility"""
    print("🚩 Auto-flagging using win32api...")
    
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * grid_width + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Flagging mine at ({click_x}, {click_y})")
        
        if click_at_position_win32(click_x, click_y, right_click=True):
            print(f"  ✓ Flag successful")
            break  # Only flag one per call
        else:
            print(f"  ✗ Flag failed")
    
    time.sleep(0.2)

def auto_click_tiles_pyautogui(tiles, safe_tiles, grid_width):
    """Alternative auto-click using PyAutoGUI for better reliability"""
    print("🟩 Auto-clicking safe tiles using PyAutoGUI...")
    
    # Move mouse to safe position first
    pyautogui.moveTo(794, 296)
    time.sleep(0.1)
    
    for (row, col) in safe_tiles:
        if row * grid_width + col >= len(tiles):
            continue
            
        x, y, w, h = tiles[row * grid_width + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Clicking safe tile at ({click_x}, {click_y})")
        
        # Use PyAutoGUI for more reliable clicking
        try:
            pyautogui.click(click_x, click_y)
            time.sleep(0.1)  # Wait for game to process
        except Exception as e:
            print(f"Failed to click at ({click_x}, {click_y}): {e}")
    
    # Move back to safe position
    pyautogui.moveTo(794, 296)

def auto_flag_tiles_pyautogui(tiles, mine_tiles, grid_width, grid, gray, cropped_bgr):
    """Alternative auto-flag using PyAutoGUI"""
    print("🚩 Auto-flagging using PyAutoGUI...")
    
    # Move mouse to safe position first
    pyautogui.moveTo(794, 296)
    time.sleep(0.1)
    
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * grid_width + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Flagging mine at ({click_x}, {click_y})")
        
        try:
            pyautogui.rightClick(click_x, click_y)
            time.sleep(0.2)  # Wait for game to process
            break  # Only flag one per call
        except Exception as e:
            print(f"Failed to flag at ({click_x}, {click_y}): {e}")
    
    pyautogui.moveTo(794, 296)

def auto_click_tiles_pynput(tiles, safe_tiles, grid_width):
    """Auto-click using pynput for Roblox compatibility"""
    print("🟩 Auto-clicking safe tiles using pynput...")
    
    for (row, col) in safe_tiles:
        if row * grid_width + col >= len(tiles):
            continue
            
        x, y, w, h = tiles[row * grid_width + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Clicking safe tile at ({click_x}, {click_y})")
        
        if click_at_position_pynput(click_x, click_y, right_click=False):
            print(f"  ✓ Click successful")
        else:
            print(f"  ✗ Click failed")
        
        time.sleep(0.2)  # Longer delay for Roblox

def auto_flag_tiles_pynput(tiles, mine_tiles, grid_width, grid, gray, cropped_bgr):
    """Auto-flag using pynput for Roblox compatibility"""
    print("🚩 Auto-flagging using pynput...")
    
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * grid_width + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Flagging mine at ({click_x}, {click_y})")
        
        if click_at_position_pynput(click_x, click_y, right_click=True):
            print(f"  ✓ Flag successful")
            break  # Only flag one per call
        else:
            print(f"  ✗ Flag failed")
    
    time.sleep(0.3)

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
        
        # Add debug output
        print(f"Attempting to flag tile at ({click_x}, {click_y}) - jittered to ({jittered_x}, {jittered_y})")
        
        success = safe_mouse_click_retry(jittered_x, jittered_y, right=True)
        if not success:
            print(f"Failed to click tile at ({jittered_x}, {jittered_y})")
            continue
            
        time.sleep(0.2)  # Increased delay for game UI update

        # Re-classify this tile and its neighbors with fresh image
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
        
        # Add debug output
        print(f"Attempting to click safe tile at ({click_x}, {click_y}) - jittered to ({jittered_x}, {jittered_y})")
        
        success = safe_mouse_click_retry(jittered_x, jittered_y, right=False)
        if not success:
            print(f"Failed to click tile at ({jittered_x}, {jittered_y})")
        time.sleep(0.1)  # Increased delay between clicks
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
    if not grid or not grid[0]:  # Check if grid is empty
        return []
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
    if not grid or not grid[0]:  # Check if grid is empty
        return 0
    return sum(1 for r, c in get_neighbors(row, col, grid) if grid[r][c] == value)

def get_unknown_neighbors(row, col, grid):
    if not grid or not grid[0]:  # Check if grid is empty
        return []
    rows, cols = len(grid), len(grid[0])
    return [(r, c) for r, c in get_neighbors(row, col, grid) 
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == "?"]

def solver_step_optimized(grid):
    """Optimized solver with better logic and efficiency"""
    safe, mines = set(), set()
    
    # Track which tiles we've already analyzed to avoid redundant work
    analyzed_tiles = set()
    
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val in "12345678":
                n = int(val)
                flagged = count_neighbors(r, c, grid, "F")
                unknown = count_neighbors(r, c, grid, "?")
                unknown_neighbors = get_unknown_neighbors(r, c, grid)
                
                # Skip if we've already analyzed this tile
                tile_key = (r, c)
                if tile_key in analyzed_tiles:
                    continue
                
                # Basic logic: if all mines are flagged, remaining are safe
                if n == flagged and unknown > 0:
                    safe.update(unknown_neighbors)
                    analyzed_tiles.add(tile_key)
                
                # If remaining unknown equals remaining mines, all are mines
                elif n - flagged == unknown and unknown > 0:
                    mines.update(unknown_neighbors)
                    analyzed_tiles.add(tile_key)
    
    return safe, mines

## Key Pattern Recognition Features:

#1. **1-2 Pattern**: When a 1 and 2 share exactly one unknown tile, and the 1 has no flagged neighbors. The unknown tile to the 2 must be a mine.
#2. **1-1 Pattern**: When two 1s share exactly one unknown tile, and both have no flagged neighbors. The shared tile must be safe.
#3. **2-2 Pattern**: When two 2s share exactly two unknown tiles, and both have no flagged neighbors. Both shared tiles must be mines.
#4. **Island Analysis**: Analyze isolated groups of unknown tiles to find patterns.
#5. **Constraint Satisfaction**: Use a simplified constraint satisfaction approach to find forced moves.

def pattern_recognition_solver(grid):
    """Advanced pattern recognition for complex Minesweeper situations"""
    safe, mines = set(), set()
    
    print("🧠 Starting pattern recognition solver...")
    
    # Pattern 1: 4-pattern (very common and powerful)
    print("\n--- Pattern 1: 4-Pattern Detection ---")
    safe_4, mines_4 = detect_4_pattern(grid)
    safe.update(safe_4)
    mines.update(mines_4)
    
    # Pattern 2: Constraint-based patterns
    print("\n--- Pattern 2: Constraint Pattern Detection ---")
    safe_constraint, mines_constraint = detect_constraint_patterns(grid)
    safe.update(safe_constraint)
    mines.update(mines_constraint)
    
    # Pattern 3: Basic 1-2 Pattern
    print("\n--- Pattern 3: Basic 1-2 Pattern Detection ---")
    safe_basic_1_2, mines_basic_1_2 = detect_1_2_pattern(grid)
    safe.update(safe_basic_1_2)
    mines.update(mines_basic_1_2)
    
    # Pattern 4: Advanced 1-2 Pattern
    print("\n--- Pattern 4: Advanced 1-2 Pattern Detection ---")
    safe_1_2, mines_1_2 = detect_advanced_1_2_pattern(grid)
    safe.update(safe_1_2)
    mines.update(mines_1_2)
    
    # Pattern 5: 1-1 Pattern
    print("\n--- Pattern 5: 1-1 Pattern Detection ---")
    safe_1_1, mines_1_1 = detect_1_1_pattern(grid)
    safe.update(safe_1_1)
    mines.update(mines_1_1)
    
    # Pattern 6: 2-2 Pattern
    print("\n--- Pattern 6: 2-2 Pattern Detection ---")
    safe_2_2, mines_2_2 = detect_2_2_pattern(grid)
    safe.update(safe_2_2)
    mines.update(mines_2_2)
    
    print(f"\n🎯 Pattern recognition complete: {len(safe)} safe, {len(mines)} mines found")
    
    return safe, mines

def detect_4_pattern(grid):
    """Detect when a 4 has exactly 3 unknown neighbors and 1 flagged neighbor"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:  # Check if grid is empty
        return safe, mines
    
    print("🔍 Scanning for 4-patterns...")
    
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "4":
                flagged = count_neighbors(r, c, grid, "F")
                unknown = get_unknown_neighbors(r, c, grid)
                
                print(f"  Found 4 at ({r}, {c}): {flagged} flagged, {len(unknown)} unknown")
                
                # 4 pattern: 1 flagged + 3 unknown = 4 total
                if flagged == 1 and len(unknown) == 3:
                    print(f" Found 4-pattern at ({r}, {c}): all 3 unknown neighbors must be mines")
                    print(f"    Unknown neighbors: {unknown}")
                    mines.update(unknown)
                elif flagged == 1:
                    print(f"    ⚠️ 4 at ({r}, {c}) has {flagged} flagged but {len(unknown)} unknown (not 3)")
                elif len(unknown) == 3:
                    print(f"    ⚠️ 4 at ({r}, {c}) has {len(unknown)} unknown but {flagged} flagged (not 1)")
    
    if mines:
        print(f"✅ 4-pattern found {len(mines)} mines to flag")
    else:
        print("❌ No 4-patterns found")
    
    return safe, mines

def detect_constraint_patterns(grid):
    """Detect patterns based on constraint satisfaction"""
    safe, mines = set(), set()
    
    print("🔍 Scanning for constraint patterns...")
    
    # Look for tiles that must be mines based on surrounding constraints
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "?":
                # Check if this tile must be a mine based on surrounding numbers
                must_be_mine = check_must_be_mine(r, c, grid)
                if must_be_mine:
                    print(f" Tile at ({r}, {c}) must be a mine based on constraints")
                    mines.add((r, c))
                
                # Check if this tile must be safe based on surrounding numbers
                must_be_safe = check_must_be_safe(r, c, grid)
                if must_be_safe:
                    print(f" Tile at ({r}, {c}) must be safe based on constraints")
                    safe.add((r, c))
    
    if safe or mines:
        print(f"✅ Constraint patterns found: {len(safe)} safe, {len(mines)} mines")
    else:
        print("❌ No constraint patterns found")
    
    return safe, mines

def check_must_be_mine(row, col, grid):
    """Check if a tile must be a mine based on surrounding constraints"""
    # Check all neighboring numbered tiles
    for nr, nc in get_neighbors(row, col, grid):
        if grid[nr][nc] in "12345678":
            n = int(grid[nr][nc])
            flagged = count_neighbors(nr, nc, grid, "F")
            unknown = get_unknown_neighbors(nr, nc, grid)
            
            # If this tile is the only unknown neighbor and the number needs exactly one more mine
            if len(unknown) == 1 and n - flagged == 1:
                print(f"    Found constraint: {n} at ({nr}, {nc}) needs 1 more mine, only 1 unknown")
                return True
    
    return False

def check_must_be_safe(row, col, grid):
    """Check if a tile must be safe based on surrounding constraints"""
    # Check all neighboring numbered tiles
    for nr, nc in get_neighbors(row, col, grid):
        if grid[nr][nc] in "12345678":
            n = int(grid[nr][nc])
            flagged = count_neighbors(nr, nc, grid, "F")
            unknown = get_unknown_neighbors(nr, nc, grid)
            
            # If this tile is the only unknown neighbor and all mines are already flagged
            if len(unknown) == 1 and n == flagged:
                print(f"    Found constraint: {n} at ({nr}, {nc}) has all mines flagged, only 1 unknown")
                return True
    
    return False

def detect_1_2_pattern(grid):
    """Detect 1-2 pattern: when a 1 and 2 share exactly one unknown tile"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:  # Check if grid is empty
        return safe, mines
    
    for r1, row1 in enumerate(grid):
        for c1, val1 in enumerate(row1):
            if val1 == "1":
                # Find all unknown neighbors of this 1
                unknown_1 = get_unknown_neighbors(r1, c1, grid)
                flagged_1 = count_neighbors(r1, c1, grid, "F")
                
                # If 1 has exactly 1 unknown neighbor, it must be safe
                if len(unknown_1) == 1 and flagged_1 == 0:
                    safe.update(unknown_1)
                    continue
                
                # Look for 2s that share unknowns with this 1
                for r2, row2 in enumerate(grid):
                    for c2, val2 in enumerate(row2):
                        if val2 == "2" and (r1, c1) != (r2, c2):
                            unknown_2 = get_unknown_neighbors(r2, c2, grid)
                            flagged_2 = count_neighbors(r2, c2, grid, "F")
                            
                            # Find shared unknowns
                            shared = set(unknown_1) & set(unknown_2)
                            unique_to_1 = set(unknown_1) - set(unknown_2)
                            unique_to_2 = set(unknown_2) - set(unknown_1)
                            
                            # 1-2 pattern: 1 has 1 unknown, 2 has 2 unknowns, they share 1
                            if (len(unknown_1) == 1 and len(unknown_2) == 2 and 
                                len(shared) == 1 and flagged_1 == 0 and flagged_2 == 0):
                                # The unique tile to 2 must be a mine
                                mines.update(unique_to_2)
                                # The shared tile must be safe
                                safe.update(shared)
    
    return safe, mines

def detect_1_1_pattern(grid):
    """Detect 1-1 pattern: when two 1s share exactly one unknown tile"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:  # Check if grid is empty
        return safe, mines
    
    for r1, row1 in enumerate(grid):
        for c1, val1 in enumerate(row1):
            if val1 == "1":
                unknown_1 = get_unknown_neighbors(r1, c1, grid)
                flagged_1 = count_neighbors(r1, c1, grid, "F")
                
                for r2, row2 in enumerate(grid):
                    for c2, val2 in enumerate(row2):
                        if val2 == "1" and (r1, c1) != (r2, c2):
                            unknown_2 = get_unknown_neighbors(r2, c2, grid)
                            flagged_2 = count_neighbors(r2, c2, grid, "F")
                            
                            shared = set(unknown_1) & set(unknown_2)
                            unique_to_1 = set(unknown_1) - set(unknown_2)
                            unique_to_2 = set(unknown_2) - set(unknown_1)
                            
                            # 1-1 pattern: both have 1 unknown, they share 1
                            if (len(unknown_1) == 1 and len(unknown_2) == 1 and 
                                len(shared) == 1 and flagged_1 == 0 and flagged_2 == 0):
                                # The shared tile must be safe (both 1s need it)
                                safe.update(shared)
    
    return safe, mines

def detect_2_2_pattern(grid):
    """Detect 2-2 pattern: when two 2s share exactly two unknown tiles"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:  # Check if grid is empty
        return safe, mines
    
    for r1, row1 in enumerate(grid):
        for c1, val1 in enumerate(row1):
            if val1 == "2":
                unknown_1 = get_unknown_neighbors(r1, c1, grid)
                flagged_1 = count_neighbors(r1, c1, grid, "F")
                
                for r2, row2 in enumerate(grid):
                    for c2, val2 in enumerate(row2):
                        if val2 == "2" and (r1, c1) != (r2, c2):
                            unknown_2 = get_unknown_neighbors(r2, c2, grid)
                            flagged_2 = count_neighbors(r2, c2, grid, "F")
                            
                            shared = set(unknown_1) & set(unknown_2)
                            unique_to_1 = set(unknown_1) - set(unknown_2)
                            unique_to_2 = set(unknown_2) - set(unknown_1)
                            
                            # 2-2 pattern: both have 2 unknowns, they share 2
                            if (len(unknown_1) == 2 and len(unknown_2) == 2 and 
                                len(shared) == 2 and flagged_1 == 0 and flagged_2 == 0):
                                # Both shared tiles must be mines
                                mines.update(shared)
    
    return safe, mines

def analyze_islands(grid):
    """Analyze isolated groups of unknown tiles"""
    safe, mines = set(), set()
    
    # Find all unknown tiles
    unknown_tiles = set()
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "?":
                unknown_tiles.add((r, c))
    
    # Group unknown tiles into islands (connected components)
    islands = []
    visited = set()
    
    for tile in unknown_tiles:
        if tile not in visited:
            island = set()
            stack = [tile]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    island.add(current)
                    # Add connected unknown neighbors
                    for nr, nc in get_neighbors(current[0], current[1], grid):
                        if grid[nr][nc] == "?" and (nr, nc) not in visited:
                            stack.append((nr, nc))
            islands.append(island)
    
    # Analyze each island
    for island in islands:
        island_safe, island_mines = analyze_single_island(island, grid)
        safe.update(island_safe)
        mines.update(island_mines)
    
    return safe, mines

def analyze_single_island(island, grid):
    """Analyze a single island of connected unknown tiles"""
    safe, mines = set(), set()
    
    # Find all numbered tiles that border this island
    border_numbers = set()
    for r, c in island:
        for nr, nc in get_neighbors(r, c, grid):
            if grid[nr][nc] in "12345678":
                border_numbers.add((nr, nc))
    
    # If island is small enough, try all possible mine configurations
    if len(island) <= 4:  # Only for small islands
        island_list = list(island)
        total_mines = sum(int(grid[nr][nc]) - count_neighbors(nr, nc, grid, "F") 
                        for nr, nc in border_numbers)
        
        # Try different mine placements
        valid_configs = []
        for mine_count in range(min(total_mines + 1, len(island) + 1)):
            # This is a simplified version - in practice you'd need more sophisticated logic
            pass
    
    return safe, mines

def constraint_satisfaction(grid):
    """Use constraint satisfaction to find forced moves"""
    safe, mines = set(), set()
    
    # Build a system of equations based on numbered tiles
    constraints = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val in "12345678":
                n = int(val)
                flagged = count_neighbors(r, c, grid, "F")
                unknown = get_unknown_neighbors(r, c, grid)
                if unknown:
                    constraints.append((unknown, n - flagged))
    
    # Try to solve the constraint system
    # This is a simplified version - full implementation would be more complex
    if len(constraints) <= 6:  # Only for small constraint systems
        # Try different combinations
        pass
    
    return safe, mines

def detect_advanced_1_2_pattern(grid):
    """Advanced 1-2 pattern detection for complex situations"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:  # Check if grid is empty
        return safe, mines
    
    for r1, row1 in enumerate(grid):
        for c1, val1 in enumerate(row1):
            if val1 == "1":
                unknown_1 = get_unknown_neighbors(r1, c1, grid)
                flagged_1 = count_neighbors(r1, c1, grid, "F")
                
                # Look for 2s that share unknowns with this 1
                for r2, row2 in enumerate(grid):
                    for c2, val2 in enumerate(row2):
                        if val2 == "2" and (r1, c1) != (r2, c2):
                            unknown_2 = get_unknown_neighbors(r2, c2, grid)
                            flagged_2 = count_neighbors(r2, c2, grid, "F")
                            
                            # Find shared and unique unknowns
                            shared = set(unknown_1) & set(unknown_2)
                            unique_to_1 = set(unknown_1) - set(unknown_2)
                            unique_to_2 = set(unknown_2) - set(unknown_1)
                            
                            # Advanced 1-2 pattern: 1 has 2 unknowns, 2 has 3 unknowns, they share 2
                            if (len(unknown_1) == 2 and len(unknown_2) == 3 and 
                                len(shared) == 2 and flagged_1 == 0 and flagged_2 == 0):
                                # The unique tile to 2 must be a mine
                                if unique_to_2:
                                    print(f" Found advanced 1-2 pattern: unique tile to 2 must be mine")
                                    mines.update(unique_to_2)
    
    return safe, mines

def analyze_edge_patterns(grid):
    """Analyze edge and corner patterns for additional moves"""
    safe, mines = set(), set()
    rows, cols = len(grid), len(grid[0])
    
    # Check corners first (they have fewer neighbors)
    corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    for r, c in corners:
        if grid[r][c] in "12345678":
            n = int(grid[r][c])
            flagged = count_neighbors(r, c, grid, "F")
            unknown = count_neighbors(r, c, grid, "?")
            
            # Corner tiles with 1 neighbor are often solvable
            if unknown == 1 and n - flagged == 1:
                unknown_neighbors = get_unknown_neighbors(r, c, grid)
                mines.update(unknown_neighbors)
            elif unknown == 1 and n == flagged:
                unknown_neighbors = get_unknown_neighbors(r, c, grid)
                safe.update(unknown_neighbors)
    
    # Check edges for patterns
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid[r][c] in "12345678":
                n = int(grid[r][c])
                flagged = count_neighbors(r, c, grid, "F")
                unknown = count_neighbors(r, c, grid, "?")
                unknown_neighbors = get_unknown_neighbors(r, c, grid)
                
                # Edge tiles with 2 neighbors often have forced moves
                if unknown == 2:
                    if n - flagged == 2:
                        mines.update(unknown_neighbors)
                    elif n == flagged:
                        safe.update(unknown_neighbors)
    
    return safe, mines

def probability_analysis(grid):
    """Use probability analysis when no forced moves are available"""
    safe, mines = set(), set()
    
    # Find tiles with lowest mine probability
    min_prob = float('inf')
    best_tile = None
    
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "?":
                # Calculate mine probability based on surrounding numbers
                prob = calculate_mine_probability(r, c, grid)
                if prob < min_prob:
                    min_prob = prob
                    best_tile = (r, c)
    
    # If we found a very low probability tile, suggest it as safe
    if best_tile and min_prob < 0.3:  # Less than 30% chance of mine
        safe.add(best_tile)
    
    return safe, mines

def calculate_mine_probability(row, col, grid):
    """Calculate the probability that a tile contains a mine"""
    total_constraints = 0
    total_mines_needed = 0
    
    # Check all neighboring numbered tiles
    for nr, nc in get_neighbors(row, col, grid):
        if grid[nr][nc] in "12345678":
            n = int(grid[nr][nc])
            flagged = count_neighbors(nr, nc, grid, "F")
            unknown = count_neighbors(nr, nc, grid, "?")
            
            if unknown > 0:
                total_constraints += 1
                total_mines_needed += (n - flagged)
    
    if total_constraints == 0:
        return 0.5  # Default probability if no constraints
    
    # Average probability based on surrounding constraints
    avg_prob = total_mines_needed / total_constraints
    return min(avg_prob, 1.0)  # Cap at 100%

def detect_advanced_edge_patterns(grid):
    """Analyze edge and corner patterns for additional moves"""
    safe, mines = set(), set()
    rows, cols = len(grid), len(grid[0])
    
    # Check corners first (they have fewer neighbors)
    corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    for r, c in corners:
        if grid[r][c] in "12345678":
            n = int(grid[r][c])
            flagged = count_neighbors(r, c, grid, "F")
            unknown = count_neighbors(r, c, grid, "?")
            
            # Corner tiles with 1 neighbor are often solvable
            if unknown == 1 and n - flagged == 1:
                unknown_neighbors = get_unknown_neighbors(r, c, grid)
                mines.update(unknown_neighbors)
            elif unknown == 1 and n == flagged:
                unknown_neighbors = get_unknown_neighbors(r, c, grid)
                safe.update(unknown_neighbors)
    
    # Check edges for patterns
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid[r][c] in "12345678":
                n = int(grid[r][c])
                flagged = count_neighbors(r, c, grid, "F")
                unknown = count_neighbors(r, c, grid, "?")
                unknown_neighbors = get_unknown_neighbors(r, c, grid)
                
                # Edge tiles with 2 neighbors often have forced moves
                if unknown == 2:
                    if n - flagged == 2:
                        mines.update(unknown_neighbors)
                    elif n == flagged:
                        safe.update(unknown_neighbors)
    
    return safe, mines

def advanced_pattern_solver(grid):
    """Advanced pattern solver with reliable pattern detection"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    print("🧠 Advanced pattern analysis...")
    
    # Pattern 1: Advanced 1-2-1 pattern
    safe_1_2_1, mines_1_2_1 = detect_1_2_1_pattern(grid)
    safe.update(safe_1_2_1)
    mines.update(mines_1_2_1)
    
    # Pattern 2: Advanced 2-1-2 pattern  
    safe_2_1_2, mines_2_1_2 = detect_2_1_2_pattern(grid)
    safe.update(safe_2_1_2)
    mines.update(mines_2_1_2)
    
    # Pattern 3: Advanced 1-3-1 pattern
    safe_1_3_1, mines_1_3_1 = detect_1_3_1_pattern(grid)
    safe.update(safe_1_3_1)
    mines.update(mines_1_3_1)
    
    # Pattern 4: Advanced 2-2-2 pattern
    safe_2_2_2, mines_2_2_2 = detect_2_2_2_pattern(grid)
    safe.update(safe_2_2_2)
    mines.update(mines_2_2_2)
    
    # Pattern 5: Advanced 3-3-3 pattern
    safe_3_3_3, mines_3_3_3 = detect_3_3_3_pattern(grid)
    safe.update(safe_3_3_3)
    mines.update(mines_3_3_3)
    
    # Pattern 6: Advanced island patterns
    safe_island, mines_island = detect_advanced_island_patterns(grid)
    safe.update(safe_island)
    mines.update(mines_island)
    
    # Pattern 7: Advanced edge patterns
    safe_edge, mines_edge = detect_advanced_edge_patterns(grid)
    safe.update(safe_edge)
    mines.update(mines_edge)
    
    # Pattern 8: Enhanced diagonal patterns (more reliable)
    safe_diag, mines_diag = detect_diagonal_patterns_enhanced(grid)
    safe.update(safe_diag)
    mines.update(mines_diag)
    
    # Pattern 9: Simple constraint solver (more reliable)
    safe_constraint, mines_constraint = simple_constraint_solver(grid)
    safe.update(safe_constraint)
    mines.update(mines_constraint)
    
    if safe or mines:
        print(f"✅ Advanced patterns found: {len(safe)} safe, {len(mines)} mines")
    else:
        print("❌ No advanced patterns found")
    
    return safe, mines

def detect_stuck_situation(grid):
    """Detect if we're in a stuck situation and suggest best moves"""
    # Don't call solver_step here to avoid recursion
    safe, mines = set(), set()
    
    print("🤔 No forced moves available - analyzing stuck situation...")
    
    # Get all unknown tiles
    unknown_tiles = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "?":
                prob = calculate_mine_probability_improved(r, c, grid)
                unknown_tiles.append((r, c, prob))
    
    # Sort by probability (lowest first)
    unknown_tiles.sort(key=lambda x: x[2])
    
    if unknown_tiles:
        best_tile, best_prob = unknown_tiles[0][:2], unknown_tiles[0][2]
        print(f"🎯 Best guess: Tile at {best_tile} with {best_prob:.1%} mine probability")
        
        # Suggest the best tile even if probability isn't great
        if best_prob < 0.5:  # Less than 50% chance
            safe.add(best_tile)
            print(f"✅ Suggesting safe click on tile with {best_prob:.1%} mine probability")
        else:
            print(f"⚠️ All remaining tiles have high mine probability ({best_prob:.1%})")
    
    return safe, mines

def solver_step(grid):
    """Enhanced solver with reliable pattern recognition"""
    # Check if grid is valid
    if not grid or not grid[0]:
        return set(), set()
    
    # Try optimized solver first (fastest)
    safe, mines = solver_step_optimized(grid)
    if safe or mines:
        return safe, mines
    
    # Try advanced pattern recognition (more reliable)
    safe, mines = advanced_pattern_solver(grid)
    if safe or mines:
        return safe, mines
    
    # Try pattern recognition solver as fallback
    safe, mines = pattern_recognition_solver(grid)
    
    if safe or mines:
        return safe, mines
    
    # Only try stuck detection as last resort
    return detect_stuck_situation(grid)

def calculate_mine_probability_improved(row, col, grid):
    """Improved probability calculation with better heuristics"""
    total_constraints = 0
    total_mines_needed = 0
    constraint_weights = []
    
    # Check all neighboring numbered tiles
    for nr, nc in get_neighbors(row, col, grid):
        if grid[nr][nc] in "12345678":
            n = int(grid[nr][nc])
            flagged = count_neighbors(nr, nc, grid, "F")
            unknown = count_neighbors(nr, nc, grid, "?")
            
            if unknown > 0:
                # Weight constraints by how many unknown neighbors they have
                weight = 1.0 / unknown  # More weight to constraints with fewer unknowns
                total_constraints += weight
                total_mines_needed += (n - flagged) * weight
                constraint_weights.append((n - flagged, unknown))
    
    if total_constraints == 0:
        return 0.5  # Default probability if no constraints
    
    # Calculate weighted average probability
    avg_prob = total_mines_needed / total_constraints
    
    # Apply edge/corner bonuses
    rows, cols = len(grid), len(grid[0])
    if (row == 0 or row == rows-1) and (col == 0 or col == cols-1):
        # Corner tiles often have lower mine probability
        avg_prob *= 0.8
    elif row == 0 or row == rows-1 or col == 0 or col == cols-1:
        # Edge tiles have slightly lower probability
        avg_prob *= 0.9
    
    return min(avg_prob, 1.0)  # Cap at 100%

def find_best_guess(grid):
    """Fast: Find the tile with the lowest mine probability, breaking ties by constraint count."""
    if not grid or not grid[0]:
        return None
    rows, cols = len(grid), len(grid[0])
    best_tile = None
    best_prob = 1.1
    best_constraint = -1
    candidates = []
    # First pass: compute probability and constraint count for all unknowns
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != "?":
                continue
            prob = calculate_mine_probability_improved(r, c, grid)
            neighbors = get_neighbors(r, c, grid)
            constraint_count = sum(1 for nr, nc in neighbors if grid[nr][nc] in "12345678")
            candidates.append((prob, -constraint_count, r, c))
    if not candidates:
        return None
    # Sort by probability, then by highest constraint count
    candidates.sort()
    # Only analyze the top 3 if there are ties
    top_prob = candidates[0][0]
    top_candidates = [t for t in candidates if abs(t[0] - top_prob) < 1e-6]
    if len(top_candidates) == 1:
        _, _, r, c = top_candidates[0]
        return (r, c), top_prob
    # If tie, use full analysis for top 3
    best_score = float('inf')
    best_rc = None
    for _, _, r, c in top_candidates[:3]:
        analysis = analyze_tile_for_guess(r, c, grid)
        score = calculate_guess_score(r, c, analysis, grid)
        if score < best_score:
            best_score = score
            best_rc = (r, c)
    return best_rc, top_prob

def analyze_tile_for_guess(r, c, grid):
    """Comprehensive analysis of a tile for guessing"""
    if not grid or not grid[0]:
        return {}
    
    analysis = {
        'base_probability': 0.0,
        'neighbor_numbers': [],
        'edge_bonus': 0.0,
        'corner_bonus': 0.0,
        'isolation_score': 0.0,
        'constraint_count': 0,
        'total_unknowns': 0,
        'total_mines_needed': 0
    }
    
    # Get all neighboring numbers
    neighbors = get_neighbors(r, c, grid)
    number_neighbors = [(nr, nc) for nr, nc in neighbors if grid[nr][nc] in "12345678"]
    
    analysis['neighbor_numbers'] = number_neighbors
    analysis['constraint_count'] = len(number_neighbors)
    
    if not number_neighbors:
        analysis['base_probability'] = 0.5
        return analysis
    
    # Calculate base probability from constraints
    total_weight = 0
    weighted_prob = 0
    
    for nr, nc in number_neighbors:
        n = int(grid[nr][nc])
        flagged = count_neighbors(nr, nc, grid, "F")
        unknown = get_unknown_neighbors(nr, nc, grid)
        
        if unknown:
            # Weight by inverse of unknown count (fewer unknowns = more reliable)
            weight = 1.0 / len(unknown)
            prob = (n - flagged) / len(unknown)
            
            total_weight += weight
            weighted_prob += prob * weight
            
            analysis['total_unknowns'] += len(unknown)
            analysis['total_mines_needed'] += (n - flagged)
    
    if total_weight > 0:
        analysis['base_probability'] = weighted_prob / total_weight
    
    # Edge and corner bonuses
    rows, cols = len(grid), len(grid[0])
    if (r == 0 or r == rows-1) and (c == 0 or c == cols-1):
        analysis['corner_bonus'] = -0.15  # Corners are often safer
    elif r == 0 or r == rows-1 or c == 0 or c == cols-1:
        analysis['edge_bonus'] = -0.08   # Edges are slightly safer
    
    # Isolation score (tiles with fewer neighbors are often safer)
    neighbor_count = len(get_neighbors(r, c, grid))
    if neighbor_count <= 3:
        analysis['isolation_score'] = -0.1
    elif neighbor_count >= 7:
        analysis['isolation_score'] = 0.05
    
    return analysis

def calculate_guess_score(r, c, analysis, grid):
    """Calculate a comprehensive score for guessing (lower is better)"""
    if not grid or not grid[0]:
        return 100.0
    
    base_score = analysis['base_probability'] * 100
    
    # Apply bonuses
    total_bonus = (analysis['edge_bonus'] + 
                   analysis['corner_bonus'] + 
                   analysis['isolation_score']) * 100
    
    # Constraint reliability bonus
    constraint_bonus = 0
    if analysis['constraint_count'] >= 3:
        constraint_bonus = -5  # More constraints = more reliable
    elif analysis['constraint_count'] == 0:
        constraint_bonus = 10  # No constraints = less reliable
    
    # Pattern-based bonuses
    pattern_bonus = calculate_pattern_bonus(r, c, grid)
    
    # Final score
    final_score = base_score + total_bonus + constraint_bonus + pattern_bonus
    
    # Ensure score is within reasonable bounds
    return max(0, min(100, final_score))

def calculate_pattern_bonus(r, c, grid):
    """Calculate bonus based on common Minesweeper patterns"""
    if not grid or not grid[0]:
        return 0
    
    bonus = 0
    rows, cols = len(grid), len(grid[0])
    
    # Check for 1-2-1 pattern involvement
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in "12345678":
                # Check if this number is part of a 1-2-1 pattern
                if is_part_of_1_2_1_pattern(nr, nc, grid):
                    bonus -= 3  # Slightly safer if near 1-2-1 pattern
    
    # Check for edge patterns
    if r == 0 or r == rows-1 or c == 0 or c == cols-1:
        # Edge tiles with 1 neighbor are often very safe
        neighbor_count = len(get_neighbors(r, c, grid))
        if neighbor_count <= 3:
            bonus -= 5
    
    # Check for corner patterns
    if (r == 0 or r == rows-1) and (c == 0 or c == cols-1):
        # Corner tiles are often safer
        bonus -= 8
    
    return bonus

def is_part_of_1_2_1_pattern(r, c, grid):
    """Check if a number is part of a 1-2-1 pattern"""
    if not grid or not grid[0]:
        return False
    
    rows, cols = len(grid), len(grid[0])
    
    # Check horizontal 1-2-1
    if c > 0 and c < cols - 1:
        if (grid[r][c-1] == "1" and grid[r][c] == "2" and grid[r][c+1] == "1"):
            return True
    
    # Check vertical 1-2-1
    if r > 0 and r < rows - 1:
        if (grid[r-1][c] == "1" and grid[r][c] == "2" and grid[r+1][c] == "1"):
            return True
    
    return False

# Update the probability analysis function
def probability_analysis(grid):
    """Use improved probability analysis when no forced moves are available"""
    safe, mines = set(), set()
    
    # Find tiles with lowest mine probability
    min_prob = float('inf')
    best_tile = None
    
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "?":
                # Calculate mine probability based on surrounding numbers
                prob = calculate_mine_probability_improved(r, c, grid)
                if prob < min_prob:
                    min_prob = prob
                    best_tile = (r, c)
    
    # More aggressive threshold for stuck situations
    if best_tile and min_prob < 0.5:  # Less than 50% chance of mine
        safe.add(best_tile)
        print(f"🎯 Probability-based move: Tile at {best_tile} ({min_prob:.1%} mine probability)")
    
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
    if state:
        print("▶️ Detection resumed")
    else:
        print("⏸️ Detection paused")

def quit_program():
    global RUNNING
    print("❌ Quitting")
    RUNNING = False
    # Force exit after a short delay
    import sys
    import threading
    def delayed_exit():
        time.sleep(0.5)
        sys.exit(0)
    threading.Thread(target=delayed_exit, daemon=True).start()

def force_recheck_action():
    global tiles, safe_tiles, mine_tiles, grid, inactive_tiles, first_scan, prev_labels
    if not DETECTION_ENABLED:
        print("Detection paused. Cannot force recheck.")
        return
    
    print("🔄 Force rechecking all tiles...")
    
    try:
        # Reset all state variables to force a complete re-scan
        tiles = []
        safe_tiles = set()
        mine_tiles = set()
        grid = []
        inactive_tiles = set()
        first_scan = True
        prev_labels = []
        
        print("✅ State reset complete. All tiles will be reclassified on next detection cycle.")
    except Exception as e:
        print(f"❌ Error during force recheck: {e}")
        # Fallback: just reset the scan flag
        first_scan = True

def prompt_remaining_mines():
    global remaining_mines
    try:
        # Create GUI dialog that appears on top
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Make dialog appear on top
        
        # Position dialog away from the grid area (top-left of screen)
        root.geometry("+50+50")  # Position at top-left, away from grid
        
        mine_count = simpledialog.askinteger(
            "Set Remaining Mines", 
            "Enter the number of remaining mines:",
            parent=root,
            minvalue=0,
            maxvalue=999
        )
        
        root.destroy()
        
        if mine_count is not None:
            remaining_mines = mine_count
            print(f"✅ Remaining mines set to {remaining_mines}")
        else:
            print("❌ Mine count input cancelled")
    except Exception as e:
        print(f"❌ Error setting remaining mines: {e}")
        remaining_mines = None

def move_mouse_smoothly(start_x, start_y, end_x, end_y, duration=0.3):
    """Move mouse smoothly from start to end position like a human would"""
    steps = int(duration * 60)  # 60 FPS movement
    if steps < 1:
        steps = 1
    
    for i in range(steps + 1):
        # Use easing function for natural movement
        progress = i / steps
        # Ease-out curve (starts fast, ends slow)
        eased_progress = 1 - (1 - progress) ** 3
        
        current_x = start_x + (end_x - start_x) * eased_progress
        current_y = start_y + (end_y - start_y) * eased_progress
        
        MouseController.move(int(current_x), int(current_y))
        time.sleep(duration / steps)

def auto_flag_ctypes_human():
    """Auto-flag with human-like mouse movement"""
    global tiles, mine_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not mine_tiles or not grid:
        print("Cannot auto-flag: detection disabled or no mine tiles found")
        return
    
    print("🚩 Auto-flagging mines with human-like movement...")
    
    # Get current mouse position
    import win32api
    current_x, current_y = win32api.GetCursorPos()
    
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Flagging mine at ({click_x}, {click_y})")
        
        try:
            # Move mouse smoothly to target
            move_mouse_smoothly(current_x, current_y, click_x, click_y, duration=0.4)
            time.sleep(0.1)  # Brief pause at target
            
            # Right click
            MouseController.click(click_x, click_y, right=True)
            time.sleep(0.3)  # Wait for game to process
            
            print(f"  ✓ Flagged successfully")
            current_x, current_y = click_x, click_y  # Update current position
            break  # Only flag one per call
            
        except Exception as e:
            print(f"  ✗ Failed to flag: {e}")
    
    # Move back to safe position smoothly
    move_mouse_smoothly(current_x, current_y, 794, 296, duration=0.3)
    print("Auto-flag complete")

def auto_click_ctypes_human():
    """Auto-click with human-like mouse movement"""
    global tiles, safe_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not safe_tiles or not grid:
        print("Cannot auto-click: detection disabled or no safe tiles found")
        return
    
    print(" Auto-clicking safe tiles with human-like movement...")
    
    # Get current mouse position
    import win32api
    current_x, current_y = win32api.GetCursorPos()
    
    for (row, col) in safe_tiles:
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Clicking safe tile at ({click_x}, {click_y})")
        
        try:
            # Move mouse smoothly to target
            move_mouse_smoothly(current_x, current_y, click_x, click_y, duration=0.4)
            time.sleep(0.1)  # Brief pause at target
            
            # Left click
            MouseController.click(click_x, click_y, right=False)
            time.sleep(0.25)  # Wait for game to process
            
            print(f"  ✓ Clicked successfully")
            current_x, current_y = click_x, click_y  # Update current position
            
        except Exception as e:
            print(f"  ✗ Failed to click: {e}")
    
    # Move back to safe position smoothly
    move_mouse_smoothly(current_x, current_y, 794, 296, duration=0.3)
    print("Auto-click complete")

def move_mouse_human_like(start_x, start_y, end_x, end_y, duration=0.5):
    """Move mouse with slight human-like imperfections"""
    steps = int(duration * 60)
    if steps < 1:
        steps = 1
    
    # Add slight curve to movement (not perfectly straight)
    control_x = start_x + (end_x - start_x) * 0.5 + random.randint(-10, 10)
    control_y = start_y + (end_y - start_y) * 0.5 + random.randint(-10, 10)
    
    for i in range(steps + 1):
        progress = i / steps
        
        # Quadratic Bezier curve for natural movement
        t = progress
        current_x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
        current_y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
        
        # Add slight jitter
        jitter_x = random.randint(-1, 1)
        jitter_y = random.randint(-1, 1)
        
        MouseController.move(int(current_x + jitter_x), int(current_y + jitter_y))
        time.sleep(duration / steps)

def auto_flag_human_advanced():
    """Auto-flag with advanced human-like movement"""
    global tiles, mine_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not mine_tiles or not grid:
        print("Cannot auto-flag: detection disabled or no mine tiles found")
        return
    
    print("🚩 Auto-flagging mines with advanced human-like movement...")
    
    import win32api
    current_x, current_y = win32api.GetCursorPos()
    
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        
        print(f"Flagging mine at ({click_x}, {click_y})")
        
        try:
            # Move with human-like imperfections
            move_mouse_human_like(current_x, current_y, click_x, click_y, duration=0.6)
            time.sleep(0.15)  # Pause like a human would
            
            # Right click
            MouseController.click(click_x, click_y, right=True)
            time.sleep(0.3)
            
            print(f"  ✓ Flagged successfully")
            current_x, current_y = click_x, click_y
            break
            
        except Exception as e:
            print(f"  ✗ Failed to flag: {e}")
    
    # Move back naturally
    move_mouse_human_like(current_x, current_y, 794, 296, duration=0.4)
    print("Auto-flag complete")
    
def move_mouse_smoothly_fast(start_x, start_y, end_x, end_y, duration=0.2):
    """Faster mouse movement with fewer steps"""
    steps = int(duration * 30)  # Reduced from 60 to 30 FPS
    if steps < 1:
        steps = 1
    
    for i in range(steps + 1):
        progress = i / steps
        # Faster easing curve
        eased_progress = 1 - (1 - progress) ** 2  # Changed from ^3 to ^2
        
        current_x = start_x + (end_x - start_x) * eased_progress
        current_y = start_y + (end_y - start_y) * eased_progress
        
        MouseController.move(int(current_x), int(current_y))
        time.sleep(duration / steps)

def auto_flag_optimized():
    """Flowing auto-flag: instantly jump just outside each mine tile, then smoothly move onto it and click"""
    global tiles, mine_tiles, grid
    if not DETECTION_ENABLED or not tiles or not mine_tiles or not grid:
        return
    current_x, current_y = get_mouse_position()
    mine_targets = []
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        mine_targets.append((row, col, x, y, w, h, click_x, click_y))
    # Greedy nearest neighbor path
    path = []
    visited = set()
    pos_x, pos_y = current_x, current_y
    while len(visited) < len(mine_targets):
        best_idx = None
        best_dist = float('inf')
        for i, (row, col, x, y, w, h, tx, ty) in enumerate(mine_targets):
            if i in visited:
                continue
            dist = ((tx - pos_x) ** 2 + (ty - pos_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is None:
            break
        visited.add(best_idx)
        path.append(mine_targets[best_idx])
        pos_x, pos_y = mine_targets[best_idx][6], mine_targets[best_idx][7]
    # For each tile: jump just outside, then move in and click
    for (row, col, x, y, w, h, click_x, click_y) in path:
        # Pick a random angle and radius just outside the tile
        angle = random.uniform(0, 2 * np.pi)
        radius = max(w, h) // 2 + random.randint(20, 40)
        offset_x = int(click_x + radius * np.cos(angle))
        offset_y = int(click_y + radius * np.sin(angle))
        # Instantly move to the offset point
        MouseController.move(offset_x, offset_y)
        # Smoothly move onto the tile center
        move_mouse_ultra_smooth(offset_x, offset_y, click_x, click_y, duration=0.07)
        MouseController.click(click_x, click_y, right=True)
        # Minimal pause
        time.sleep(0.005)

def auto_click_optimized():
    """Flowing auto-click: instantly jump just outside each safe tile, then smoothly move onto it and click"""
    global tiles, safe_tiles, grid
    if not DETECTION_ENABLED or not tiles or not safe_tiles or not grid:
        return
    current_x, current_y = get_mouse_position()
    safe_targets = []
    for (row, col) in safe_tiles:
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        safe_targets.append((row, col, x, y, w, h, click_x, click_y))
    # Greedy nearest neighbor path
    path = []
    visited = set()
    pos_x, pos_y = current_x, current_y
    while len(visited) < len(safe_targets):
        best_idx = None
        best_dist = float('inf')
        for i, (row, col, x, y, w, h, tx, ty) in enumerate(safe_targets):
            if i in visited:
                continue
            dist = ((tx - pos_x) ** 2 + (ty - pos_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is None:
            break
        visited.add(best_idx)
        path.append(safe_targets[best_idx])
        pos_x, pos_y = safe_targets[best_idx][6], safe_targets[best_idx][7]
    # For each tile: jump just outside, then move in and click
    for (row, col, x, y, w, h, click_x, click_y) in path:
        angle = random.uniform(0, 2 * np.pi)
        radius = max(w, h) // 2 + random.randint(20, 40)
        offset_x = int(click_x + radius * np.cos(angle))
        offset_y = int(click_y + radius * np.sin(angle))
        MouseController.move(offset_x, offset_y)
        move_mouse_ultra_smooth(offset_x, offset_y, click_x, click_y, duration=0.06)
        MouseController.click(click_x, click_y, right=False)
        time.sleep(0.005)

def auto_flag_batch():
    """Batch flag multiple mines at once"""
    global tiles, mine_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not mine_tiles or not grid:
        print("Cannot auto-flag: detection disabled or no mine tiles found")
        return
    
    print("🚩 Batch flagging mines...")
    
    import win32api
    current_x, current_y = win32api.GetCursorPos()
    
    # Get all mine positions
    mine_positions = []
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        mine_positions.append((row, col, click_x, click_y))
    
    # Flag up to 4 mines quickly (smaller batch for speed)
    max_flags = min(4, len(mine_positions))
    
    for i, (row, col, click_x, click_y) in enumerate(mine_positions[:max_flags]):
        print(f"Flagging mine {i+1}/{max_flags} at ({click_x}, {click_y})")
        
        try:
            # Quick movement
            move_mouse_smoothly_fast(current_x, current_y, click_x, click_y, duration=0.2)
            time.sleep(0.03)
            
            # Right click
            MouseController.click(click_x, click_y, right=True)
            time.sleep(0.1)  # Very short wait
            
            print(f"  ✓ Flagged successfully")
            current_x, current_y = click_x, click_y
            
        except Exception as e:
            print(f"  ✗ Failed to flag: {e}")
    
    print("Batch flagging complete")

def move_mouse_ultra_smooth(start_x, start_y, end_x, end_y, duration=0.15):
    """Ultra-smooth mouse movement with high precision"""
    # For very short distances, use direct movement
    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
    if distance < 10:  # Very short distance
        MouseController.move(int(end_x), int(end_y))
        time.sleep(0.01)  # Brief pause for precision
        return
    
    # For longer distances, use smooth movement with higher precision
    fps = 120  # Higher FPS for smoother movement
    steps = max(1, int(duration * fps))
    
    for i in range(steps + 1):
        progress = i / steps
        
        # Smooth easing curve with better precision
        eased_progress = progress * progress * (3 - 2 * progress)  # Smoothstep
        
        current_x = start_x + (end_x - start_x) * eased_progress
        current_y = start_y + (end_y - start_y) * eased_progress
        
        MouseController.move(int(current_x), int(current_y))
        time.sleep(duration / steps)
    
    # Final precision adjustment
    MouseController.move(int(end_x), int(end_y))
    time.sleep(0.01)  # Brief pause before clicking

def move_mouse_with_deceleration(start_x, start_y, end_x, end_y, duration=0.2):
    """Move mouse with smooth deceleration curve"""
    fps = 60
    steps = int(duration * fps)
    if steps < 1:
        steps = 1
    
    for i in range(steps + 1):
        progress = i / steps
        
        # Deceleration curve (starts fast, ends slow)
        eased_progress = 1 - (1 - progress) ** 2
        
        current_x = start_x + (end_x - start_x) * eased_progress
        current_y = start_y + (end_y - start_y) * eased_progress
        
        MouseController.move(int(current_x), int(current_y))
        time.sleep(duration / steps)

def auto_flag_ultra_fast():
    """Ultra-fast auto-flag with smooth movement"""
    global tiles, mine_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not mine_tiles or not grid:
        debug_print("Cannot auto-flag: detection disabled or no mine tiles found")
        return
    
    debug_print("🚩 Ultra-fast auto-flagging with smooth movement...")
    
    current_x, current_y = get_mouse_position()
    
    # Process all mines quickly
    mine_positions = []
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        mine_positions.append((click_x, click_y))
    
    # Flag all mines with smooth movement
    for i, (click_x, click_y) in enumerate(mine_positions):
        debug_print(f"Flagging mine {i+1}/{len(mine_positions)} at ({click_x}, {click_y})")
        
        try:
            # Move to target with smooth deceleration
            move_mouse_with_deceleration(current_x, current_y, click_x, click_y, duration=0.12)
            MouseController.click(click_x, click_y, right=True)
            time.sleep(0.05)  # Brief pause for game to process
            
            debug_print(f"  ✓ Flagged successfully")
            current_x, current_y = click_x, click_y
            
        except Exception as e:
            debug_print(f"  ✗ Failed to flag: {e}")
    
    debug_print("Ultra-fast auto-flag complete")

def auto_click_ultra_fast():
    """Ultra-fast auto-click with smooth movement"""
    global tiles, safe_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not safe_tiles or not grid:
        debug_print("Cannot auto-click: detection disabled or no safe tiles found")
        return
    
    debug_print(" Ultra-fast auto-clicking with smooth movement...")
    
    current_x, current_y = get_mouse_position()
    
    # Process all safe tiles quickly
    safe_positions = []
    for (row, col) in safe_tiles:
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        safe_positions.append((click_x, click_y))
    
    # Click all safe tiles with smooth movement
    for i, (click_x, click_y) in enumerate(safe_positions):
        debug_print(f"Clicking safe tile {i+1}/{len(safe_positions)} at ({click_x}, {click_y})")
        
        try:
            # Move to target with smooth deceleration
            move_mouse_with_deceleration(current_x, current_y, click_x, click_y, duration=0.1)
            MouseController.click(click_x, click_y, right=False)
            time.sleep(0.03)  # Brief pause for game to process
            
            debug_print(f"  ✓ Clicked successfully")
            current_x, current_y = click_x, click_y
            
        except Exception as e:
            debug_print(f"  ✗ Failed to click: {e}")
    
    debug_print("Ultra-fast auto-click complete")

def auto_flag_batch():
    """Batch flag with smooth movement"""
    global tiles, mine_tiles, grid
    
    if not DETECTION_ENABLED or not tiles or not mine_tiles or not grid:
        print("Cannot auto-flag: detection disabled or no mine tiles found")
        return
    
    print("🚩 Batch flagging mines with smooth movement...")
    
    current_x, current_y = get_mouse_position()
    
    # Get all mine positions
    mine_positions = []
    for (row, col) in mine_tiles:
        if grid[row][col] == "F":
            continue
            
        x, y, w, h = tiles[row * len(grid[0]) + col]
        click_x, click_y = X1 + x + w // 2, Y1 + y + h // 2
        mine_positions.append((click_x, click_y))
    
    # Flag up to 4 mines quickly with smooth movement
    max_flags = min(4, len(mine_positions))
    
    for i, (click_x, click_y) in enumerate(mine_positions[:max_flags]):
        print(f"Flagging mine {i+1}/{max_flags} at ({click_x}, {click_y})")
        
        try:
            # Move to target with smooth deceleration
            move_mouse_with_deceleration(current_x, current_y, click_x, click_y, duration=0.15)
            MouseController.click(click_x, click_y, right=True)
            time.sleep(0.05)  # Brief pause for game to process
            
            print(f"  ✓ Flagged successfully")
            current_x, current_y = click_x, click_y
            
        except Exception as e:
            print(f"  ✗ Failed to flag: {e}")
    
    print("Batch flagging complete")

def move_mouse_with_click_during_movement(start_x, start_y, click_x, click_y, end_x, end_y, duration=0.25, right_click=True):
    """Move mouse continuously from start to end, clicking at the click point during movement"""
    fps = 120
    steps = int(duration * fps)
    if steps < 1:
        steps = 1
    
    # Calculate when to click (about 60% through the movement)
    click_step = int(steps * 0.6)
    click_performed = False
    
    for i in range(steps + 1):
        progress = i / steps
        
        # Use deceleration curve
        eased_progress = 1 - (1 - progress) ** 2
        
        # Interpolate between start and end
        current_x = start_x + (end_x - start_x) * eased_progress
        current_y = start_y + (end_y - start_y) * eased_progress
        
        MouseController.move(int(current_x), int(current_y))
        
        # Click when we're close to the click target
        if i >= click_step and not click_performed:
            # Check if we're close enough to the click target
            distance_to_click = ((current_x - click_x) ** 2 + (current_y - click_y) ** 2) ** 0.5
            if distance_to_click < 15:  # Within 15 pixels
                # Click at current mouse position, not target position
                MouseController.click(int(current_x), int(current_y), right=right_click)
                click_performed = True
        
        time.sleep(duration / steps)

def detect_1_2_1_pattern(grid):
    """Detect 1-2-1 pattern: 1-?-2-?-1"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    rows, cols = len(grid), len(grid[0])
    
    # Check horizontal 1-2-1 pattern
    for r in range(rows):
        for c in range(cols - 4):
            if (grid[r][c] == "1" and grid[r][c+2] == "2" and grid[r][c+4] == "1" and
                grid[r][c+1] == "?" and grid[r][c+3] == "?"):
                
                # Check if 1s have no flagged neighbors
                flagged_1a = count_neighbors(r, c, grid, "F")
                flagged_1b = count_neighbors(r, c+4, grid, "F")
                
                if flagged_1a == 0 and flagged_1b == 0:
                    # The shared unknown must be safe, the other must be a mine
                    safe.add((r, c+1))  # Shared unknown
                    mines.add((r, c+3))  # Unique to 2
    
    return safe, mines

def detect_2_1_2_pattern(grid):
    """Detect 2-1-2 pattern: 2-?-1-?-2"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    rows, cols = len(grid), len(grid[0])
    
    # Check horizontal 2-1-2 pattern
    for r in range(rows):
        for c in range(cols - 4):
            if (grid[r][c] == "2" and grid[r][c+2] == "1" and grid[r][c+4] == "2" and
                grid[r][c+1] == "?" and grid[r][c+3] == "?"):
                
                flagged_2a = count_neighbors(r, c, grid, "F")
                flagged_2b = count_neighbors(r, c+4, grid, "F")
                
                if flagged_2a == 0 and flagged_2b == 0:
                    # Both unknowns must be mines
                    mines.add((r, c+1))
                    mines.add((r, c+3))
    
    return safe, mines

def detect_1_3_1_pattern(grid):
    """Detect 1-3-1 pattern: 1-?-?-3-?-?-1"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    rows, cols = len(grid), len(grid[0])
    
    # Check horizontal 1-3-1 pattern
    for r in range(rows):
        for c in range(cols - 6):
            if (grid[r][c] == "1" and grid[r][c+3] == "3" and grid[r][c+6] == "1" and
                grid[r][c+1] == "?" and grid[r][c+2] == "?" and 
                grid[r][c+4] == "?" and grid[r][c+5] == "?"):
                
                flagged_1a = count_neighbors(r, c, grid, "F")
                flagged_1b = count_neighbors(r, c+6, grid, "F")
                
                if flagged_1a == 0 and flagged_1b == 0:
                    # The two shared unknowns must be safe
                    safe.add((r, c+1))
                    safe.add((r, c+2))
                    # The two unique to 3 must be mines
                    mines.add((r, c+4))
                    mines.add((r, c+5))
    
    return safe, mines

def detect_2_2_2_pattern(grid):
    """Detect 2-2-2 pattern: 2-?-2-?-2"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    rows, cols = len(grid), len(grid[0])
    
    # Check horizontal 2-2-2 pattern
    for r in range(rows):
        for c in range(cols - 4):
            if (grid[r][c] == "2" and grid[r][c+2] == "2" and grid[r][c+4] == "2" and
                grid[r][c+1] == "?" and grid[r][c+3] == "?"):
                
                flagged_2a = count_neighbors(r, c, grid, "F")
                flagged_2b = count_neighbors(r, c+2, grid, "F")
                flagged_2c = count_neighbors(r, c+4, grid, "F")
                
                if flagged_2a == 0 and flagged_2b == 0 and flagged_2c == 0:
                    # Both unknowns must be mines
                    mines.add((r, c+1))
                    mines.add((r, c+3))
    
    return safe, mines

def detect_3_3_3_pattern(grid):
    """Detect 3-3-3 pattern: 3-?-?-3-?-?-3"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    rows, cols = len(grid), len(grid[0])
    
    # Check horizontal 3-3-3 pattern
    for r in range(rows):
        for c in range(cols - 6):
            if (grid[r][c] == "3" and grid[r][c+3] == "3" and grid[r][c+6] == "3" and
                grid[r][c+1] == "?" and grid[r][c+2] == "?" and 
                grid[r][c+4] == "?" and grid[r][c+5] == "?"):
                
                flagged_3a = count_neighbors(r, c, grid, "F")
                flagged_3b = count_neighbors(r, c+3, grid, "F")
                flagged_3c = count_neighbors(r, c+6, grid, "F")
                
                if flagged_3a == 0 and flagged_3b == 0 and flagged_3c == 0:
                    # All four unknowns must be mines
                    mines.add((r, c+1))
                    mines.add((r, c+2))
                    mines.add((r, c+4))
                    mines.add((r, c+5))
    
    return safe, mines

def detect_advanced_island_patterns(grid):
    """Detect advanced island patterns"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    # Find isolated groups of unknown tiles
    unknown_tiles = set()
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "?":
                unknown_tiles.add((r, c))
    
    # Group unknown tiles into islands
    islands = []
    visited = set()
    
    for tile in unknown_tiles:
        if tile not in visited:
            island = set()
            stack = [tile]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    island.add(current)
                    # Add connected unknown neighbors
                    for nr, nc in get_neighbors(current[0], current[1], grid):
                        if grid[nr][nc] == "?" and (nr, nc) not in visited:
                            stack.append((nr, nc))
            islands.append(island)
    
    # Analyze each island
    for island in islands:
        island_safe, island_mines = analyze_island_constraints(island, grid)
        safe.update(island_safe)
        mines.update(island_mines)
    
    return safe, mines

def analyze_island_constraints(island, grid):
    """Analyze constraints for a single island"""
    safe, mines = set(), set()
    
    # Find all numbered tiles that border this island
    border_numbers = set()
    for r, c in island:
        for nr, nc in get_neighbors(r, c, grid):
            if grid[nr][nc] in "12345678":
                border_numbers.add((nr, nc))
    
    # If island is small enough, try constraint satisfaction
    if len(island) <= 4 and len(border_numbers) <= 6:
        island_list = list(island)
        
        # Build constraint system
        constraints = []
        for nr, nc in border_numbers:
            n = int(grid[nr][nc])
            flagged = count_neighbors(nr, nc, grid, "F")
            island_neighbors = [i for i, (r, c) in enumerate(island_list) 
                              if (r, c) in get_neighbors(nr, nc, grid)]
            if island_neighbors:
                constraints.append((island_neighbors, n - flagged))
        
        # Try different mine placements
        for mine_count in range(len(island) + 1):
            for mine_positions in itertools.combinations(range(len(island)), mine_count):
                mine_set = set(mine_positions)
                
                # Check if this configuration satisfies all constraints
                valid = True
                for island_indices, required_mines in constraints:
                    actual_mines = len(set(island_indices) & mine_set)
                    if actual_mines != required_mines:
                        valid = False
                        break
                
                if valid:
                    # Found valid configuration
                    for i in range(len(island_list)):
                        if i in mine_set:
                            mines.add(island_list[i])
                        else:
                            safe.add(island_list[i])
                    return safe, mines
    
    return safe, mines

def detect_diagonal_patterns_enhanced(grid):
    """Detect reliable diagonal patterns with strict validation"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    rows, cols = len(grid), len(grid[0])
    
    # Only the most reliable diagonal pattern: 1-2-1 with perfect isolation
    for r in range(rows - 2):
        for c in range(cols - 2):
            # Check if we have a perfect 1-2-1 diagonal pattern
            if (grid[r][c] == "1" and grid[r+1][c+1] == "2" and grid[r+2][c+2] == "1" and
                grid[r][c+1] == "?" and grid[r+1][c] == "?" and 
                grid[r+1][c+2] == "?" and grid[r+2][c+1] == "?"):
                
                # Strict validation: check that no other numbers affect these tiles
                flagged_1a = count_neighbors(r, c, grid, "F")
                flagged_2 = count_neighbors(r+1, c+1, grid, "F")
                flagged_1b = count_neighbors(r+2, c+2, grid, "F")
                
                # Only proceed if all numbers have exactly the right number of flags
                if (flagged_1a == 0 and flagged_2 == 0 and flagged_1b == 0):
                    # Additional check: ensure no other numbers are adjacent to our unknowns
                    unknowns = [(r, c+1), (r+1, c), (r+1, c+2), (r+2, c+1)]
                    isolated = True
                    
                    for ur, uc in unknowns:
                        for nr, nc in get_neighbors(ur, uc, grid):
                            if grid[nr][nc] in "12345678" and (nr, nc) not in [(r, c), (r+1, c+1), (r+2, c+2)]:
                                isolated = False
                                break
                        if not isolated:
                            break
                    
                    if isolated:
                        # This is a reliable pattern
                        safe.add((r, c+1))  # Shared between 1s
                        safe.add((r+2, c+1))  # Shared between 1s
                        mines.add((r+1, c))   # Unique to 2
                        mines.add((r+1, c+2)) # Unique to 2
    
    return safe, mines



def simple_constraint_solver(grid):
    """Simple but reliable constraint solver for small groups"""
    safe, mines = set(), set()
    
    if not grid or not grid[0]:
        return safe, mines
    
    # Find small isolated groups of unknowns (2-4 tiles)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "?":
                # Get connected unknown tiles
                connected = get_connected_unknowns(r, c, grid)
                if 2 <= len(connected) <= 4:
                    # Check if this group is isolated (no other numbers affect it)
                    if is_group_isolated(connected, grid):
                        result = solve_small_group(connected, grid)
                        if result:
                            group_safe, group_mines = result
                            safe.update(group_safe)
                            mines.update(group_mines)
    
    return safe, mines

def get_connected_unknowns(r, c, grid):
    """Get all unknown tiles connected to (r,c)"""
    if not grid or not grid[0]:
        return set()
    
    visited = set()
    to_visit = [(r, c)]
    connected = set()
    
    while to_visit:
        curr_r, curr_c = to_visit.pop()
        if (curr_r, curr_c) in visited or grid[curr_r][curr_c] != "?":
            continue
            
        visited.add((curr_r, curr_c))
        connected.add((curr_r, curr_c))
        
        # Add adjacent unknowns
        for nr, nc in get_neighbors(curr_r, curr_c, grid):
            if grid[nr][nc] == "?" and (nr, nc) not in visited:
                to_visit.append((nr, nc))
    
    return connected

def is_group_isolated(group, grid):
    """Check if a group of unknowns is isolated from other numbers"""
    if not grid or not grid[0]:
        return False
    
    rows, cols = len(grid), len(grid[0])
    
    # Get all numbers that could affect this group
    affecting_numbers = set()
    for r, c in group:
        for nr, nc in get_neighbors(r, c, grid):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in "12345678":
                affecting_numbers.add((nr, nc))
    
    # Check if any of these numbers have unknowns outside our group
    for nr, nc in affecting_numbers:
        for nnr, nnc in get_neighbors(nr, nc, grid):
            if (nnr, nnc) not in group and grid[nnr][nnc] == "?":
                return False
    
    return True

def solve_small_group(group, grid):
    """Solve a small isolated group of unknowns"""
    if not group or len(group) > 4:
        return None
    
    # Build constraints for this group
    constraints = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val in "12345678":
                n = int(val)
                flagged = count_neighbors(r, c, grid, "F")
                unknown = get_unknown_neighbors(r, c, grid)
                # Only include constraints that affect our group
                group_unknowns = [pos for pos in unknown if pos in group]
                if group_unknowns:
                    constraints.append((group_unknowns, n - flagged))
    
    if not constraints:
        return None
    
    # Try all possible mine placements in this group
    group_list = list(group)
    for mine_count in range(len(group_list) + 1):
        for mine_positions in itertools.combinations(group_list, mine_count):
            mine_set = set(mine_positions)
            
            # Check if this configuration satisfies all constraints
            valid = True
            for unknown_list, required_mines in constraints:
                actual_mines = len(set(unknown_list) & mine_set)
                if actual_mines != required_mines:
                    valid = False
                    break
            
            if valid:
                # Found valid configuration
                safe_tiles = group - mine_set
                return safe_tiles, mine_set
    
    return None

def find_best_hint_tile(grid):
    """Find the '?' tile that, if revealed, would provide the most information to the solver."""
    if not grid or not grid[0]:
        return None
    best_tile = None
    best_score = -1
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val != "?":
                continue
            neighbors = get_neighbors(r, c, grid)
            numbered_neighbors = [(nr, nc) for nr, nc in neighbors if grid[nr][nc] in "12345678"]
            constraint_count = len(numbered_neighbors)
            diversity = len(set(grid[nr][nc] for nr, nc in numbered_neighbors))
            unknown_neighbors = [(nr, nc) for nr, nc in neighbors if grid[nr][nc] == "?"]
            cluster_center = len(unknown_neighbors)
            # Weighted score
            score = (
                1.0 * constraint_count +
                0.5 * diversity +
                0.3 * cluster_center
            )
            if score > best_score:
                best_score = score
                best_tile = (r, c)
    return best_tile

def grid_to_mrgris_format(grid):
    """Convert the current grid to mrgris.com text format."""
    if not grid or not grid[0]:
        return ""
    
    # Simple compact format for easy copying
    lines = []
    for row in grid:
        line = []
        for val in row:
            if val in "12345678":
                line.append(val)
            elif val == "F":
                line.append("F")
            else:
                line.append(".")
        lines.append("".join(line))
    
    return "\n".join(lines)

def copy_to_clipboard(text):
    """Copy text to clipboard using tkinter (cross-platform)."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()  # now it stays on the clipboard after the window is closed
        root.destroy()
        print("\n📋 Board copied to clipboard!")
        print("Format: " + text.replace('\n', '\\n'))
        print("\nTo use in mrgris.com:")
        print("1. Go to: https://mrgris.com/projects/minesweepr/demo/analyzer/")
        print("2. Set board size to match your game")
        print("3. Use the mouse/keyboard to recreate the board:")
        print("   - Click on a number (0-8), then click on the board")
        print("   - Use 'f' key to flag mines")
        print("   - Use 'space' for blank/unknown cells")
        print("4. Or manually type the board from the format above")
    except Exception as e:
        print(f"Failed to copy to clipboard: {e}")

def grid_to_mrgris_url(grid):
    """Convert the current grid to mrgris.com URL format."""
    if not grid or not grid[0]:
        return None
    
    width = len(grid[0])
    height = len(grid)
    
    # Auto-detect mine count based on board size
    if width == 16 and height == 16:
        auto_mine_count = 50
    elif width == 30 and height == 16:
        auto_mine_count = 99
    elif width == 30 and height == 30:
        auto_mine_count = 200
    else:
        auto_mine_count = None
    
    # Ask user for mine count via GUI
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Make dialog appear on top
    
    # Position dialog away from the grid area
    root.geometry("+50+50")  # Position at top-left, away from grid
    
    if auto_mine_count:
        prompt_text = f"Enter the total number of mines for this {width}x{height} board (auto-detected: {auto_mine_count}):"
        default_value = auto_mine_count
    else:
        prompt_text = f"Enter the total number of mines for this {width}x{height} board:"
        default_value = 0
    
    mine_count = simpledialog.askinteger(
        "Mine Count", 
        prompt_text,
        parent=root,
        minvalue=0,
        maxvalue=width * height,
        initialvalue=default_value
    )
    
    root.destroy()
    
    if mine_count is None:  # User cancelled
        return None
    
    # Convert grid to board format
    # Numbers stay as numbers, known empty tiles (KETs) become '.', unknown tiles become 'x', flags become '!'
    board_rows = []
    for row in grid:
        row_str = ""
        for val in row:
            if val in "12345678":
                row_str += val
            elif val == "F":
                row_str += "!"
            elif val == "0":  # Known empty tile (KET)
                row_str += "."
            elif val == "?":  # Unknown tile
                row_str += "x"
            else:  # Fallback for any other values
                row_str += "x"
        board_rows.append(row_str)
    
    # Join rows with colons
    board = ":".join(board_rows)
    
    # Build URL
    url = f"https://mrgris.com/projects/minesweepr/demo/analyzer/?w={width}&h={height}&mines={mine_count}&board={board}"
    
    return url

def export_board_if_no_forced_moves():
    global grid, safe_tiles, mine_tiles
    if grid and grid[0]:
        board_str = grid_to_mrgris_format(grid)
        copy_to_clipboard(board_str)
        
        # Create and open URL with board data
        url = grid_to_mrgris_url(grid)
        if url:
            try:
                webbrowser.open(url)
                print(f"\n🌐 Opening mrgris.com with your board pre-loaded!")
                print(f"URL: {url}")
            except Exception as e:
                print(f"Failed to open browser: {e}")
        else:
            print("Export cancelled by user or failed to create board URL")
    else:
        print("Cannot export: Board is empty or not detected.")

def export_board_if_no_forced_moves_threadsafe():
    threading.Thread(target=export_board_if_no_forced_moves, daemon=True).start()

def main():
    global DETECTION_ENABLED, RUNNING, tiles, safe_tiles, mine_tiles, grid, remaining_mines, inactive_tiles, first_scan, gray, cropped_bgr, FULL_AUTO_MODE

    # Setup hotkeys
    setup_hotkeys()

    width, height = X2 - X1, Y2 - Y1

    cv.namedWindow("Detected Tiles + Solver Hints", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Detected Tiles + Solver Hints", cv.WND_PROP_TOPMOST, 1)
    cv.resizeWindow("Detected Tiles + Solver Hints", width, height)

    prev_labels = []
    first_scan = True
    full_auto_thread = None
    
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
            if tiles and labels:  # Only process if we have tiles and labels
                xs = sorted(set(x for (x, y, w, h) in tiles))
                ys = sorted(set(y for (x, y, w, h) in tiles))
                x_map = {x: i for i, x in enumerate(xs)}
                y_map = {y: i for i, y in enumerate(ys)}
                for idx, (x, y, w, h) in enumerate(tiles):
                    if idx < len(labels):  # Safety check for labels
                        row = y_map[y]
                        col = x_map[x]
                        if idx in inactive_tiles:
                            # Draw orange for inactive/skipped tiles
                            cv.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                        else:
                            label = labels[idx]
                            cv.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            cv.putText(debug_img, label, (x + 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Only draw safe/mine tiles if grid is valid
            if grid and grid[0]:
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
                
                # Best guess (blue box) - always show when no forced moves
                if not safe_tiles and not mine_tiles:
                    best_guess_result = find_best_guess(grid)
                    if best_guess_result:
                        best_tile, best_prob = best_guess_result
                        if best_tile and best_tile[0] * len(grid[0]) + best_tile[1] < len(tiles):
                            x, y, w, h = tiles[best_tile[0] * len(grid[0]) + best_tile[1]]
                            
                            # Get detailed analysis for display
                            analysis = analyze_tile_for_guess(best_tile[0], best_tile[1], grid)
                            
                            # Draw blue box for best guess
                            cv.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                            
                            # Show detailed information
                            prob_text = f"BEST {int(best_prob*100)}%"
                            cv.putText(debug_img, prob_text, (x + 2, y + h - 20),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                            
                            # Show constraint count
                            constraint_text = f"C:{analysis['constraint_count']}"
                            cv.putText(debug_img, constraint_text, (x + 2, y + h - 8),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                            
                            # Show edge/corner indicator
                            if analysis['corner_bonus'] < 0:
                                cv.putText(debug_img, "CORNER", (x + w - 35, y + 8),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                            elif analysis['edge_bonus'] < 0:
                                cv.putText(debug_img, "EDGE", (x + w - 25, y + 8),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    # --- Improved Best hint (purple box) ---
                    best_hint = find_best_hint_tile(grid)
                    if best_hint and best_hint[0] * len(grid[0]) + best_hint[1] < len(tiles):
                        x, y, w, h = tiles[best_hint[0] * len(grid[0]) + best_hint[1]]
                        cv.rectangle(debug_img, (x, y), (x + w, y + h), (180, 0, 255), 3)
                        cv.putText(debug_img, "HINT", (x + 2, y + h - 4),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 255), 2)

            cv.imshow("Detected Tiles + Solver Hints", debug_img)

            # Process any queued actions (hotkeys)
            process_action_queue()

            # Check if full auto mode should start/stop
            if FULL_AUTO_MODE and (mine_tiles or safe_tiles):
                if full_auto_thread is None or not full_auto_thread.is_alive():
                    import threading
                    full_auto_thread = threading.Thread(target=run_full_auto, daemon=True)
                    full_auto_thread.start()
            elif not FULL_AUTO_MODE and full_auto_thread and full_auto_thread.is_alive():
                # Full auto was disabled, thread will stop on next cycle
                pass

        else:
            time.sleep(CHECK_DELAY)

        if cv.waitKey(1) & 0xFF == 27:
            break

        elapsed = time.time() - start_time
        if elapsed < CHECK_DELAY:
            time.sleep(CHECK_DELAY - elapsed)

    cv.destroyAllWindows()

def setup_hotkeys():
    """Setup hotkeys with different speed options"""
    try:
        # Clear any existing hotkeys to avoid conflicts
        keyboard.unhook_all()
        
        # Setup all hotkeys with proper suppression
        keyboard.add_hotkey('alt+p', lambda: set_detection(False), suppress=True)
        keyboard.add_hotkey('alt+r', lambda: set_detection(True), suppress=True)
        keyboard.add_hotkey('alt+q', quit_program, suppress=True)
        keyboard.add_hotkey('alt+f', lambda: queue_action('flag_flow'), suppress=True)
        keyboard.add_hotkey('alt+k', lambda: queue_action('click_flow'), suppress=True)
        keyboard.add_hotkey('alt+n', toggle_full_auto, suppress=True)
        keyboard.add_hotkey('ctrl+r', force_recheck_action, suppress=True)
        keyboard.add_hotkey('ctrl+m', prompt_remaining_mines, suppress=True)
        
        # Export hotkey - run in a separate thread to avoid blocking
        keyboard.add_hotkey('alt+l', export_board_if_no_forced_moves_threadsafe, suppress=True)
        
        print("✅ All hotkeys configured successfully")
    except Exception as e:
        print(f"❌ Error setting up hotkeys: {e}")

if __name__ == "__main__":
    print("🎮 Minesweeper Solver with Online Export")
    print("=" * 50)
    print("Hotkeys:")
    print("  Alt+P = Pause detection")
    print("  Alt+R = Resume detection") 
    print("  Alt+Q = Quit program")
    print("  Alt+F = Auto-flag mines (continuous)")
    print("  Alt+K = Auto-click safe tiles (continuous)")
    print("  Alt+N = Toggle full auto mode")
    print("  Ctrl+R = Force recheck board")
    print("  Ctrl+M = Set remaining mines")
    print("  Alt+L = Export board to mrgris.com")
    print("=" * 50)
    main()
