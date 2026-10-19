"""Goal-graph data structures + JSON serialization.

A GoalGraph is a sequence of (pose, image) nodes the robot follows to reach
a far goal in image-conditioned mode. Written once per run (at data-gen /
rollout time) into the run dir as:

  <run_dir>/frames.json   — manifest (GoalGraph.load reads this)
  <run_dir>/frames/       — sibling PNG dir: 000.png, 001.png, ...

The SubgoalSelector consumes a loaded graph at rollout time and returns the
current subgoal image based on the robot's pose.

Closest-node estimator: pose-based. OmniVLA's deployment recipe uses image
embeddings (NoMaD/ViNT-style); we substitute Euclidean pose distance because
sim provides ground-truth pose. For real-world use, swap SubgoalSelector for
an embedding-based one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class GraphNode:
    xy: tuple[float, float]
    yaw_rad: float
    image_filename: str   # basename inside the graph's image_dir
    # Latest (v, w) controller command at sample time. Bonus telemetry — not
    # used as a training label (actions are derived from inter-node pose
    # deltas, as in upstream OmniVLA's GNM dataset). Useful for sanity
    # checks (does ∫(v,w)dt ≈ node delta?) and failure analysis.
    cmd_v: float | None = None    # m/s, forward velocity command
    cmd_w: float | None = None    # rad/s, yaw rate command


@dataclass
class GoalGraph:
    env_config_name: str
    env_config_hash: str           # truncated sha1 of env_config.to_dict(); detects drift
    node_spacing_m: float
    robot_inflation_m: float
    # Goal-object geometry. Recorded so downstream consumers (success
    # detection, pose-mode goal pose) can reuse the same "where to stop"
    # semantics. Defaults keep JSONs without these fields loadable.
    goal_radius_m: float = 0.0     # max XY half-extent of /World/Goal AABB
    standoff_m: float = 0.0        # distance the A* terminus is short of object centre
    # How the nodes were sampled, e.g.:
    #   "static_arclength" — teleport-and-snap at uniform `node_spacing_m`
    #                        spacing along the smoothed A* path.
    #   "policy_rollout"   — closed-loop drive sampled at `sample_hz`;
    #                        node_spacing_m is 0 (spacing follows the motion).
    sampling_mode: str = "static_arclength"
    target_speed_mps: float = 0.0
    sample_hz: float = 0.0
    nodes: list[GraphNode] = field(default_factory=list)

    @staticmethod
    def image_dir_for(json_path: Path) -> Path:
        return Path(json_path).with_suffix("")

    def save(self, json_path: Path) -> None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        # asdict turns the tuple xy into a list — JSON handles either, fine
        json_path.write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, json_path: Path) -> "GoalGraph":
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
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class SubgoalSelector:
    """Position-based forward-only node tracker for a GoalGraph.

    Algorithm (OmniVLA's topological-graph deployment):
        1. Estimate the *current* location as the closest node (within a
           forward window — never picks a node behind us).
        2. Hand the policy the image from `look_ahead` nodes further along
           as the goal. With look_ahead=1 (default), this is the "next
           node" rule from the paper.

    Why the look-ahead matters: if we showed the current-location image
    as the goal, the policy would see "you're already here" and stop.
    The whole point of a graph is that the goal image is always a step
    ahead, pulling the robot forward.

    Last-node clamp: when `_idx + look_ahead` exceeds the graph length,
    the selector returns the final node's image — which is the
    actually-the-goal scene by construction (last A* node is at the
    standoff, facing the goal object).
    """

    def __init__(
        self,
        graph: GoalGraph,
        image_dir: Path,
        advance_window: int = 5,
        look_ahead: int = 1,
    ) -> None:
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
        self._idx = 0

    @property
    def current_index(self) -> int:
        """Index of the node the robot is *at* (closest-so-far)."""
        return self._idx

    @property
    def target_index(self) -> int:
        """Index of the node the robot is *headed toward* (current + look_ahead)."""
        return min(self._idx + self.look_ahead, len(self._images) - 1)

    @property
    def num_nodes(self) -> int:
        return len(self.graph.nodes)

    def current_image(self) -> Image.Image:
        return self._images[self.target_index]

    def current_node(self) -> GraphNode:
        return self.graph.nodes[self.target_index]

    def update(self, robot_xy: tuple[float, float]) -> None:
        best_idx = self._idx
        best_d2 = _sq_dist(robot_xy, self.graph.nodes[self._idx].xy)
        end = min(self._idx + self.advance_window, len(self.graph.nodes) - 1)
        for j in range(self._idx + 1, end + 1):
            d2 = _sq_dist(robot_xy, self.graph.nodes[j].xy)
            if d2 < best_d2:
                best_idx = j
                best_d2 = d2
        self._idx = best_idx
