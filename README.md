# Minesweeper Solver & Auto-Player with Online Export

This project is an advanced Minesweeper assistant and auto-player for Windows. It uses computer vision to detect the Minesweeper board, applies advanced solving logic and pattern recognition, and can automatically click or flag tiles using human-like mouse movement. **NEW: Export boards to online solvers when no forced moves exist!**

## Features
- **Computer Vision Detection**: Detects Minesweeper board and tiles using OpenCV
- **Advanced Solver**: Pattern recognition, probability analysis, and constraint satisfaction
- **Auto-Player**: Auto-click and auto-flag with smooth, human-like mouse movement
- **Visual Overlay**: Real-time solver hints with colored boxes and probability percentages
- **Online Export**: Export boards to [mrgris.com Minesweepr Analyzer](https://mrgris.com/projects/minesweepr/demo/analyzer/) when stuck
- **GUI Integration**: Clean dialogs for mine count input and board export
- **Hotkey Control**: Full keyboard control for all features

## Hotkeys
| Hotkey | Action |
|--------|--------|
| **Alt+P** | Pause detection |
| **Alt+R** | Resume detection |
| **Alt+Q** | Quit program |
| **Alt+F** | Auto-flag mines (continuous) |
| **Alt+K** | Auto-click safe tiles (continuous) |
| **Alt+N** | Toggle full auto mode |
| **Ctrl+R** | Force recheck board |
| **Ctrl+M** | Set remaining mines (GUI dialog) |
| **Alt+L** | Export board to mrgris.com (GUI dialog) |

## Visual Overlay Features
- **Green boxes**: Safe tiles to click
- **Red boxes**: Mines to flag  
- **Blue boxes**: Best guess when no forced moves
- **Purple boxes**: High-impact hint tiles
- **Probability percentages**: Mine probability for unknown tiles
- **Constraint counts**: Number of constraints affecting each tile

## Online Export Feature
When the solver can't find forced moves, press **Alt+L** to:
1. **Export board** to clipboard in readable format
2. **Open mrgris.com** with your board pre-loaded
3. **Get advanced analysis** from the online solver
4. **Find optimal moves** when local solver is stuck

### Export Format
- **Numbers (1-8)**: Stay as numbers
- **Known Empty Tiles (KETs)**: Converted to `.`
- **Unknown tiles**: Converted to `x`
- **Flagged mines**: Converted to `!`

## Requirements
**Python 3.8+ (Windows only)**

### Required Python Modules
- `opencv-python` (cv2)
- `numpy`
- `pyautogui`
- `keyboard`
- `pywin32` (for `win32api`, `win32con`, `win32gui`)
- `pynput`
- `tkinter` (GUI dialogs - usually included with Python)

### Optional (for faster screen capture)
- `mss`

### Standard Library (no need to install)
- `ctypes`, `threading`, `queue`, `itertools`, `random`, `re`, `collections`, `typing`, `time`, `os`

## Installation
Install all required modules with:

```bash
pip install opencv-python numpy pyautogui keyboard pywin32 pynput
```

For faster screen capture (optional):
```bash
pip install mss
```

## Usage
1. **Position your Minesweeper game** in the expected screen region (coordinates: 716, 317, 1203, 804)
2. **Run the solver**: `python minesweeper_hint.py`
3. **Use hotkeys** to control detection and auto-play
4. **When stuck**, press **Alt+L** to export to online solver

## Advanced Features

### Solver Capabilities
- **Pattern Recognition**: 1-2, 1-1, 2-2, 4-corner patterns
- **Island Analysis**: Analyzes isolated groups of unknown tiles
- **Constraint Satisfaction**: Advanced logical deduction
- **Probability Analysis**: Calculates mine probabilities for unknown tiles
- **Best Guess Algorithm**: Finds optimal moves when no forced moves exist

### Auto-Player Features
- **Human-like Movement**: Smooth mouse trajectories with easing
- **Multiple Click Methods**: ctypes, pynput, pyautogui support
- **Retry Logic**: Automatic retry on failed clicks
- **Speed Options**: Various movement speeds and styles

### Board Detection
- **Grid Line Detection**: Automatically finds board boundaries
- **Tile Classification**: Recognizes numbers, flags, and empty tiles
- **Template Matching**: Uses image templates for accurate classification
- **Real-time Updates**: Continuously monitors board changes

## Troubleshooting

### Common Issues
1. **Board not detected**: Ensure Minesweeper is in the correct screen region
2. **Hotkeys not working**: Run as administrator if needed
3. **Export fails**: Check internet connection for mrgris.com access
4. **GUI dialogs not appearing**: Ensure tkinter is properly installed

### Performance Tips
- Use `mss` for faster screen capture
- Close unnecessary applications to reduce CPU usage
- Adjust `CHECK_DELAY` in the script for different update frequencies

## Notes
- **Windows only**: Uses Windows-specific APIs for mouse control and window management
- **Template-based**: Place number/flag templates in the `templates/` directory
- **Classic Minesweeper**: Best results with classic Windows Minesweeper or compatible clones
- **Online Integration**: Requires internet connection for mrgris.com export feature

---
