class OccupancyGrid:
    """Spatial hash grid for fast proximity queries to enforce non-intersection."""

    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.grid = {}  # (row, col) -> list of (x, y)

    def insert(self, x, y):
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        key = (r, c)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append((x, y))

    def min_dist_sq(self, x, y, radius):
        """Return minimum squared distance to any registered point within search radius."""
        c0 = int((x - radius) / self.cell_size)
        c1 = int((x + radius) / self.cell_size)
        r0 = int((y - radius) / self.cell_size)
        r1 = int((y + radius) / self.cell_size)

        min_d2 = float('inf')
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                for (px, py) in self.grid.get((r, c), ()):
                    d2 = (x - px) ** 2 + (y - py) ** 2
                    if d2 < min_d2:
                        min_d2 = d2
        return min_d2

    def is_clear(self, x, y, min_spacing):
        """Return True if no registered point is within min_spacing."""
        min_sq = min_spacing * min_spacing
        found = self.min_dist_sq(x, y, min_spacing * 2)
        return found >= min_sq
