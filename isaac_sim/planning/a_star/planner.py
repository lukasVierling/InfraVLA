import heapq
import math
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline

_NEIGHBOURS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]


class PlanningError(RuntimeError):
    """Throws an Error that makes probelms better understandable (e.g. robot spawns in object etc.)"""
    pass


def _uniform_arclength_query(total_s, spacing_m):
    """returns x coordinate for uniform sampling given the max length"""
    s_q = np.arange(0.0, total_s, spacing_m)
    return np.concatenate([s_q, [total_s]])


class Path(ABC):
    """Abstract path class"""
    @property
    @abstractmethod
    def total_length(self) -> float:
        """Returns the total length of the path."""
        ...

    @abstractmethod
    def sample(self, spacing_m: float) -> list[tuple[float, float]]:
        """Returns a sampled path at a given sampling distance"""
        ...

    @abstractmethod
    def sample_with_heading(
        self, spacing_m: float
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """Samples path with given spacing and reutrn heading at each point"""
        ...


class SplinePath(Path):
    """Class that supports a spline interpolated path"""
    def __init__(self, csx, csy, total_s):
        """
        Intialized a spline path.
        
        Args
            csx: the x splines
            csy: the y splines
            total_s: the total length of the path 
        """
        self.csx, self.csy, self.total_s = csx, csy, total_s
        self._dcsx = csx.derivative()
        self._dcsy = csy.derivative()

    @property
    def total_length(self) -> float:
        """Returns the total length of the spline path."""
        return float(self.total_s)

    def sample(self, spacing_m):
        """Returns a unfiormly sampled path at given spacing in meters"""
        s_q = _uniform_arclength_query(self.total_s, spacing_m)
        return list(zip(self.csx(s_q).tolist(), self.csy(s_q).tolist()))

    def sample_with_heading(self, spacing_m):
        """Returns a uniformly sampled path a given spacing in meters with heading.
        
        This function samples at given spacing in meters and returns the path consisting of those points. 
        Additionally, it adds a heading direction which is derived from derivatives except for the start and end, since those derivatives are not useful.
        """
        s_q = _uniform_arclength_query(self.total_s, spacing_m)
        xs = self.csx(s_q)
        ys = self.csy(s_q)
        dx = self._dcsx(s_q)
        dy = self._dcsy(s_q)
        yaws = np.arctan2(dy, dx).tolist()
        positions = list(zip(xs.tolist(), ys.tolist()))
        if len(positions) >= 2:
            x0, y0 = positions[0]
            x1, y1 = positions[1]
            yaws[0] = math.atan2(y1 - y0, x1 - x0)
            xm, ym = positions[-2]
            xn, yn = positions[-1]
            yaws[-1] = math.atan2(yn - ym, xn - xm)
        return positions, yaws


class PolylinePath(Path):
    """A path class for a path not using splines just built from path waypoints."""
    def __init__(self, vertices):
        """Initialize a polylinepath instance."""
        pts = np.asarray(vertices, dtype=float)
        if len(pts) < 2:
            raise ValueError("PolylinePath needs at least two vertices")
        deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], deltas > 0])
        self._pts = pts[keep]
        self._deltas = deltas[deltas > 0]
        self._cum_s = np.concatenate([[0.0], np.cumsum(self._deltas)])

    @property
    def total_length(self) -> float:
        """Returns the total length of the path"""
        return float(self._cum_s[-1])

    def sample(self, spacing_m):
        """Resmaples along the path with given spacing distance."""
        s_q = _uniform_arclength_query(self.total_length, spacing_m)
        out_x = np.interp(s_q, self._cum_s, self._pts[:, 0])
        out_y = np.interp(s_q, self._cum_s, self._pts[:, 1])
        return list(zip(out_x.tolist(), out_y.tolist()))

    def sample_with_heading(self, spacing_m):
        """Samples along the path with given spacing distance in meters and returns headings at each point.
        
        This function does sampling along the path unfiorm with given spacing. 
        The headings returned are claculated based of the atan between those two points."""
        positions = self.sample(spacing_m)
        n = len(positions)
        yaws = []
        for i, (x, y) in enumerate(positions):
            if i < n - 1:
                nx, ny = positions[i + 1]
                yaws.append(math.atan2(ny - y, nx - x))
            else:
                px, py = positions[i - 1]
                yaws.append(math.atan2(y - py, x - px))
        return positions, yaws


class A_star:
    """A star planner class"""

    def __init__(self, a_star_cfg, map):
        """Initializes an A* planning instance with it's own map"""
        self.conf = a_star_cfg
        self.map = map

    def _is_traversable(self, r, c):
        """Checks whether a cell (r,c) is traversable"""
        H, W = self.map.sdf.shape
        if not (0 <= r < H and 0 <= c < W):
            return False
        return self.map.sdf[r, c] > self.conf.robot_inflation_m

    def _octile_h(self, a, b):
        """
        The heruistic function for A*.
        
        This function computes diagonal way to the end of the shorter axis and then linear distance to the end of the longer axis form there. 
        """
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return (abs(dr - dc) + math.sqrt(2) * min(dr, dc)) * self.map.resolution

    def _edge_cost(self, a, b, sdf_weighted, alpha, robot_radius):
        """Returns cost to get from a to b consdiering sdf weighing."""

        step = math.hypot(a[0] - b[0], a[1] - b[1]) * self.map.resolution
        if not sdf_weighted:
            return step
        clearance = min(self.map.sdf[a], self.map.sdf[b]) - robot_radius
        return step * (1.0 + alpha / (clearance + 1e-3))

    def _reconstruct_xy(self, came_from, end):
        """Reconstructs the optimal path walked to the goal."""
        cells = [end]
        while cells[-1] in came_from:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        return [self.map.get_world_coords(c) for c in cells]

    def get_a_star_path(self, start, goal) -> Path:
        """Returns the a (processed) star path for given start and goal through the whole pipeline."""
        raw = self._a_star_search(start, goal,
                                   sdf_weighted=self.conf.pipeline.sdf_weighted)
        if self.conf.pipeline.hill_climbing:
            raw = self._hill_climbing(raw)
        if self.conf.pipeline.cubic_spline:
            return self._cubic_spline_interpolation(raw)
        return PolylinePath(raw)

    def _a_star_search(self, start, goal, sdf_weighted: bool = True):
        """
        Calcualtes the A* path for a given start and goal.
        
        This function uses the A* algorihtm (allowing to walk in 8 directions) and uses ESDF maps.
        The algorihtm also considers robot radius to avoid collisions.

        Args:
            start: starting point coordinate
            goal: goal point coordinates
            sdf_weighted: indicator whether A* calculating should consdier SDF weigths for heursitic function
        
        Returns:
            path: path, the path from start to goal as a list of coordiantes
            Error: path planning error if something goes wrong
        """
        m = self.map
        r = self.conf.robot_inflation_m
        alpha = self.conf.alpha_clearance

        sr, sc = self.map.get_map_coords(start)
        gr, gc = self.map.get_map_coords(goal)

        if not self._is_traversable(sr, sc):
            raise PlanningError(f"start {start} inside inflated obstacle "
                                f"(sdf={m.sdf[sr, sc]:.3f} < {r})")

        if not self._is_traversable(gr, gc):
            raise PlanningError(f"goal {goal} inside inflated obstacle")

        open_heap = []
        g_score = {(sr, sc): 0.0}
        came_from = {}
        counter = 0
        heapq.heappush(open_heap, (self._octile_h((sr, sc), (gr, gc)), counter, (sr, sc)))

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur == (gr, gc):
                return self._reconstruct_xy(came_from, cur)
            for dr, dc in _NEIGHBOURS_8:
                nb = (cur[0] + dr, cur[1] + dc)
                if not self._is_traversable(*nb):
                    continue
                cost = self._edge_cost(cur, nb, sdf_weighted, alpha, r)
                tentative = g_score[cur] + cost
                if tentative < g_score.get(nb, float("inf")):
                    g_score[nb] = tentative
                    came_from[nb] = cur
                    counter += 1
                    f = tentative + self._octile_h(nb, (gr, gc))
                    heapq.heappush(open_heap, (f, counter, nb))
        raise PlanningError(f"no path from {start} to {goal}")

    def _hill_climbing(self, path):
        """Applies local greedy hill climbing on ESDF values."""
        m = self.map
        r = self.conf.robot_inflation_m
        window_m = self.conf.hill_climb_window_m
        half_cells = int(round(window_m / 2 / m.resolution))
        out = [path[0]]
        for xy in path[1:-1]:
            rr, cc = self.map.get_map_coords(xy)
            r_lo, r_hi = max(0, rr - half_cells), min(m.sdf.shape[0], rr + half_cells + 1)
            c_lo, c_hi = max(0, cc - half_cells), min(m.sdf.shape[1], cc + half_cells + 1)
            window = m.sdf[r_lo:r_hi, c_lo:c_hi]
            local_r, local_c = np.unravel_index(np.argmax(window), window.shape)
            best_rc = (r_lo + local_r, c_lo + local_c)
            if m.sdf[best_rc] > r:
                out.append(self.map.get_world_coords(best_rc))
            else:
                out.append(xy)
        out.append(path[-1])
        return out

    def _cubic_spline_interpolation(self, path):
        """Applies cubic spline interpolation (natural mode) to smoothen out the trajecory. Turn path into a SplinePath object."""
        pts = np.array(path)
        deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], deltas > 0])
        pts = pts[keep]
        deltas = deltas[deltas > 0]
        s = np.concatenate([[0.0], np.cumsum(deltas)])
        csx = CubicSpline(s, pts[:, 0], bc_type="natural")
        csy = CubicSpline(s, pts[:, 1], bc_type="natural")
        return SplinePath(csx, csy, s[-1])
