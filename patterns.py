def get_neighbors(r, c, grid):
    neighbors = []
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r+dr, c+dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbors.append((nr,nc))
    return neighbors

def unknown_cells(grid, cells):
    return [(r,c) for (r,c) in cells if grid[r][c] == '?']

def apply_pattern_line(grid, pattern, mines_indices, safe_indices):
    """
    Scan grid rows and columns for a pattern of numbers (as strings),
    and for each occurrence:
      - mines_indices: positions in pattern (0-based) that correspond to mines
      - safe_indices: positions in pattern that correspond to safe tiles.
    Returns sets of mines and safe tiles (coordinates).
    """
    safe = set()
    mines = set()
    rows = len(grid)
    cols = len(grid[0])
    plen = len(pattern)

    def check_line(line, r, c_start, vertical=False):
        for i in range(len(line) - plen + 1):
            window = line[i:i+plen]
            if window == pattern:
                # For each mine position index, add unknown neighbors of tile at that position
                for mi in mines_indices:
                    rr = r+i+mi if vertical else r
                    cc = c_start+i+mi if not vertical else c_start
                    neighbors = get_neighbors(rr, cc, grid)
                    for nr,nc in unknown_cells(grid, neighbors):
                        mines.add((nr,nc))
                # For each safe position index, add unknown neighbors of tile at that position
                for si in safe_indices:
                    rr = r+i+si if vertical else r
                    cc = c_start+i+si if not vertical else c_start
                    neighbors = get_neighbors(rr, cc, grid)
                    for nr,nc in unknown_cells(grid, neighbors):
                        safe.add((nr,nc))

    # Check rows
    for r in range(rows):
        line = [grid[r][c] for c in range(cols)]
        check_line(line, r, 0, vertical=False)

    # Check columns
    for c in range(cols):
        line = [grid[r][c] for r in range(rows)]
        check_line(line, 0, c, vertical=True)

    return safe, mines

def apply_patterns(grid):
    safe = set()
    mines = set()

    # Patterns from computronium.org basics, mines and safe tiles per pattern
    # Each pattern is a tuple: (pattern_list, mine_positions, safe_positions)
    patterns = [
        (['1', '2'],       [1], [0]),            # 1-2 Rule
        (['1', '1'],       [1], [1]),            # 1-1 Rule (second 1 is mine adjacent)
        (['1', '2', '2', '1'], [1,2], [0,3]),   # 1-2-2-1 Pattern
        (['1', '2', '1'],  [1], [0,2]),          # 1-2-1 Pattern
        (['1', '1'],       [0,1], []),            # 1-1+ Pattern, treat both 1's neighbors as mines
        (['1', '1', '1'],   [0,2], [1]),         # 1-1-1 Pattern, mines at edges, safe middle
        (['1', '2', '2', '2', '1'], [1,2,3], [0,4]), # 1-2-2-2-1 Pattern
        (['1', '2', '2', '2', '2', '1'], [1,2,3,4], [0,5]), # 1-2-2-2-2-1 Pattern
        (['1', '2', '3', '2', '1'], [2], [0,4]), # 1-2-3-2-1 Pattern
        (['1', '2', '3', '3', '2', '1'], [2,3], [0,5]), # 1-2-3-3-2-1 Pattern
    ]

    for pattern, mine_pos, safe_pos in patterns:
        s, m = apply_pattern_line(grid, pattern, mine_pos, safe_pos)
        safe.update(s)
        mines.update(m)

    return safe, mines
