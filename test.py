import cv2 as cv
import numpy as np
import pyautogui
import time
import keyboard  # pip install keyboard

# === Configuration ===
GRID_LINE_COLOR = np.array([128, 128, 129])
GRID_TOLERANCE = 10

BORDER_COLOR = np.array([30, 30, 30])
BORDER_TOLERANCE = 10

CHECK_DELAY = 5  # seconds between checks

X1, Y1, X2, Y2 = 716, 317, 1203, 804

DETECTION_ENABLED = True
RUNNING = True

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

    def cluster_lines(lines, max_gap=5, min_val=0):
        if len(lines) == 0:
            return []
        clustered = []
        group = [lines[0]]
        for x in lines[1:]:
            if x - group[-1] <= max_gap:
                group.append(x)
            else:
                avg = int(np.mean(group))
                if avg >= min_val:
                    clustered.append(avg)
                group = [x]
        avg = int(np.mean(group))
        if avg >= min_val:
            clustered.append(avg)
        return clustered

    v_lines = cluster_lines(vertical_lines, max_gap=5)
    h_lines = cluster_lines(horizontal_lines, max_gap=5)

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

def on_pause():
    global DETECTION_ENABLED
    DETECTION_ENABLED = False
    print("Detection paused (global hotkey).")

def on_resume():
    global DETECTION_ENABLED
    DETECTION_ENABLED = True
    print("Detection resumed (global hotkey).")

def on_quit():
    global RUNNING
    print("Quitting (global hotkey).")
    RUNNING = False

# Register global hotkeys
keyboard.add_hotkey('p', on_pause)
keyboard.add_hotkey('r', on_resume)
keyboard.add_hotkey('q', on_quit)

def main():
    global DETECTION_ENABLED, RUNNING
    while RUNNING:
        screenshot = np.array(pyautogui.screenshot())
        cropped = screenshot[Y1:Y2, X1:X2]
        cropped = cv.cvtColor(cropped, cv.COLOR_RGB2BGR)

        if DETECTION_ENABLED:
            v_lines, h_lines, mask = find_grid_lines_positions(cropped)
            tiles = generate_tile_coordinates(v_lines, h_lines)

            debug_img = cropped.copy()

            for x in v_lines:
                cv.line(debug_img, (x, 0), (x, debug_img.shape[0]), (255, 0, 0), 1)
            for y in h_lines:
                cv.line(debug_img, (0, y), (debug_img.shape[1], y), (255, 0, 0), 1)
            for (x, y, w, h) in tiles:
                cv.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

            cv.imshow("Grid Mask", mask)
            cv.imshow("Detected Tiles", debug_img)
        else:
            cv.imshow("Grid Mask", np.zeros_like(cropped))
            cv.imshow("Detected Tiles", np.zeros_like(cropped))

        # ESC key to quit if window is focused
        if cv.waitKey(1) & 0xFF == 27:
            break

        time.sleep(CHECK_DELAY)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
