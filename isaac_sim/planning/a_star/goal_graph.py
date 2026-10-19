from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class GraphNode:
    """Graph Node dataclass to store pose and recording and commands at that point"""
    xy: tuple[float, float]
    yaw_rad: float
    image_filename: str
    cmd_v: float | None = None
    cmd_w: float | None = None


@dataclass
class GoalGraph:
    """Goal Graph dataclass to store informaiton about how the path was calcualted as well as a list of graph nodes"""
    env_config_name: str
    env_config_hash: str
    node_spacing_m: float
    robot_inflation_m: float
    goal_radius_m: float = 0.0
    standoff_m: float = 0.0
    sampling_mode: str = "static_arclength"
    target_speed_mps: float = 0.0
    sample_hz: float = 0.0
    nodes: list[GraphNode] = field(default_factory=list)

    @staticmethod
    def image_dir_for(json_path: Path) -> Path:
        """creates an image directory"""
        return Path(json_path).with_suffix("")

    def save(self, json_path: Path) -> None:
        """Saves the goal graph at the given path."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        json_path.write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, json_path: Path) -> "GoalGraph":
        """Loads the goal graph from a given path and creates a node for each node stored there."""
        d = json.loads(Path(json_path).read_text())
        nodes_raw = d.pop("nodes")
        nodes = [
            GraphNode(
                xy=(float(n["xy"][0]), float(n["xy"][1])),
                yaw_rad=float(n["yaw_rad"]),
                image_filename=n["image_filename"],
                cmd_v=n.get("cmd_v"),
                cmd_w=n.get("cmd_w"),
            )
            for n in nodes_raw
        ]
        return cls(nodes=nodes, **d)


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculates the squared distance between two points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class SubgoalSelector:
    """Subgoal Selector to select the next subgoal within a graph"""

    def __init__(
        self,
        graph: GoalGraph,
        image_dir: Path,
        advance_window: int = 5,
        look_ahead: int = 1,
        extra_image_dirs: dict[str, Path] | None = None,
    ) -> None:
        """Initializes a Subgoal Selector instance.

        extra_image_dirs maps a view name (e.g. "cctv") to a sibling frame dir
        whose images are indexed by the SAME node index as the main graph. Each
        node's image_filename is reused as the basename, so the CCTV goal frame
        is the CCTV view at the very node the front graph selects. Missing files
        raise immediately (no silent black-goal fallback).
        """
        if not graph.nodes:
            raise ValueError("GoalGraph has no nodes")
        if look_ahead < 0:
            raise ValueError(f"look_ahead must be >= 0, got {look_ahead}")
        self.graph = graph
        self.advance_window = advance_window
        self.look_ahead = look_ahead
        self._images: list[Image.Image] = [
            Image.open(Path(image_dir) / n.image_filename).convert("RGB")
            for n in graph.nodes
        ]
        # Per-view goal frames, same node index as self._images.
        self._extra_images: dict[str, list[Image.Image]] = {}
        for name, dir_path in (extra_image_dirs or {}).items():
            dir_path = Path(dir_path)
            self._extra_images[name] = [
                Image.open(dir_path / n.image_filename).convert("RGB")
                for n in graph.nodes
            ]
        self._idx = 0

    @property
    def current_index(self) -> int:
        """Returns the current index of the subgoal slector position."""
        return self._idx

    @property
    def target_index(self) -> int:
        """Return the index of the target that is lookaehad many spots in the future"""
        return min(self._idx + self.look_ahead, len(self._images) - 1)

    @property
    def num_nodes(self) -> int:
        """Returns number of nodes in the Goal Graph."""
        return len(self.graph.nodes)

    def current_image(self) -> Image.Image:
        """Returns the image at the current position of subgoal selector"""
        return self._images[self.target_index]

    def current_image_for(self, view_name: str) -> Image.Image:
        """Returns the goal image for an extra view (e.g. "cctv") at the same
        target node the main graph is pointing at."""
        return self._extra_images[view_name][self.target_index]

    def current_node(self) -> GraphNode:
        """Returns the graph node corresponding to the position of subgoal seelctor."""
        return self.graph.nodes[self.target_index]

    def update(self, robot_xy: tuple[float, float]) -> None:
        """
        Updates the subgoal selector position.
        
        This function takes the given xy location of the robot, calculates the distance of each node to this position. 
        Then continues to walk along the path and picks the closest node as soon as stepping further will increase the distance.
        """
        best_idx = self._idx
        best_d2 = _sq_dist(robot_xy, self.graph.nodes[self._idx].xy)
        end = min(self._idx + self.advance_window, len(self.graph.nodes) - 1)
        for j in range(self._idx + 1, end + 1):
            d2 = _sq_dist(robot_xy, self.graph.nodes[j].xy)
            if d2 < best_d2:
                best_idx = j
                best_d2 = d2
        self._idx = best_idx
