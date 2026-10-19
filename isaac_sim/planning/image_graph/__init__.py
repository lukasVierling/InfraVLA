"""Image-goal graph data structure.

A topological sequence of (pose, front-camera image) nodes along a planned
path, written once per run at data-gen / rollout time and replayed by
swapping the goal image for the next subgoal as the robot advances. Used by
the training dataset loaders, which call GoalGraph.load(frames.json). Exports
only the lightweight graph classes so importing it stays cheap.

The sim-side planner has its own copy of this class in
planning/a_star/goal_graph.py.
"""

from .graph import GoalGraph, GraphNode, SubgoalSelector

__all__ = ["GoalGraph", "GraphNode", "SubgoalSelector"]
