"""Image-goal graph data structure (formerly isaac_sim/scouting/graph.py).

A topological sequence of (pose, front-camera image) nodes along a planned
path. Written once per run at data-gen / rollout time, then replayed by
swapping the goal image I_g for the next subgoal as the robot advances
(paper's deployment recipe, root.tex:301).

This package is the *shared, live* home for the GoalGraph data structure.
Its consumers are the training dataset loaders
(prismatic/vla/datasets/{isaac_sim,cctv_isaac_sim}_dataset.py), which call
GoalGraph.load(frames.json), and the legacy old_server_client/ scout
pipeline. It deliberately exports only the lightweight graph classes — no
numpy / scout-driver dependencies — so importing it stays cheap.

Note: the sim-side planner carries its own port of this class in
planning/a_star/goal_graph.py; the two are kept separate by design (not yet unified).
"""

from .graph import GoalGraph, GraphNode, SubgoalSelector

__all__ = ["GoalGraph", "GraphNode", "SubgoalSelector"]
