# `live/` — human-in-the-loop OmniVLA on the real Spot

One approved step at a time. The script computes; a human executes.

## Control boundary

**`operator_loop.py` is read-only toward the robot.** It creates ROS
subscriptions only — no publisher, no action client, no service client. It
cannot move Spot and, just as importantly, **it cannot stop Spot**.

Motion happens only when *you* paste and run the printed `ros2 action send_goal`
command in your own terminal.

| to stop... | use |
|---|---|
| the loop (no more proposals) | SPACE / `q` / ESC / Ctrl-C in the loop terminal |
| a motion already running | Ctrl-C in **your** send_goal terminal |
| everything, for real | **hardware E-Stop** |

The abort key in the loop is a convenience, not a safety device. The E-Stop is
the only real one.

## Pipeline

```
/camera/frontleft/compressed        ─┐
                                     ├─ stitch ─ onboard.png ─[ZMQ]─ OmniVLA ─ 8-waypoint chunk
/camera/frontright/compressed       ─┘                                            │
                                                                     waypoint[4] ─┤
                     artifacts on disk ── you review ── printed ros2 command ──────┘
                                                    │
                                          you run it → Spot moves
```

## Preprocessing, in order

0. **Our format** — `chunk[4] = [x, y, cos θ, sin θ]`, normalised units, body
   frame (+x forward, +y left). Index 4 is "the 5th waypoint" and is the same
   carrot the sim controller steers to.
1. **Denormalise** — `x, y × spacing_m`. Default **0.305**, the mean of GNM's
   per-dataset constants (cory_hall 0.06 … tartan_drive 0.72). It is a property
   of the training mixture, *not* of how fast we drive. `cos/sin` are a
   direction and are never scaled.
2. **Heading** — `yaw = atan2(sin, cos)` → `qz = sin(yaw/2)`, `qw = cos(yaw/2)`.
3. **Frame check** — OmniVLA (+x fwd, +y left) matches Spot body (+x fwd,
   +y left), so values pass through unchanged. Stated explicitly because a wrong
   `y` sign mirrors every turn and still looks plausible.
4. **Clamp** — kinematic feasibility cap, `max_speed_mps * (waypoint_index + 1)
   / tick_hz` (0.578 * 5 / 3 = **0.963 m** at the defaults). `wp[k]` is `k+1`
   ticks ahead, so this is the furthest the robot could physically have walked
   to reach it; beyond that the prediction is claiming a teleport, which is the
   only thing a clamp should catch. Flagged in the review output and recorded in
   `step.json` as `max_step_cap_m` / `max_step_cap_source`. Set `max_step_m > 0`
   to override with a fixed value.

   A fixed cap does not scale with the spacing: at spacing 0.19 a flat 0.75 m
   would truncate ~22% of ground-truth chunks (p90 step 0.772 m), turning the
   guard into a systematic under-step. The derived cap clamps none of them and
   rescales correctly if `waypoint_index` changes.

## Stitching

`stitch_live.py` is the BD `stitch_front_images` method driven from live ROS
data: both frames projected onto a plane 2 m ahead of a virtual camera midway
between the two real ones, using live `camera_info` intrinsics and `tf_static`
extrinsics. Built once (the rig is rigid), then two `cv2.remap` calls per frame.

Result is ~96° × 80° with a **known focal length**, which is what makes the
bearing arrow on the plot meaningful. Two deviations from BD's example, both
deliberate: the virtual eye is the *midpoint* of the two cameras (their code
sums the positions, putting the eye ~0.5 m too far forward), and the cameras are
gain-matched over their overlap before blending (they auto-expose independently;
without this the seam is a visible brightness step).

## Run

The GPU side serves the model over ZMQ. The ROS 2 side (`operator_loop.py`) needs ROS 2 with
`rclpy`, `tf2_ros`, OpenCV, matplotlib and pyzmq.

```bash
# GPU side: a merged checkpoint (see real_world/train/README.md), conda env active
CKPT=checkpoints/<run>_step3000_merged real_world/live/serve_spot_ft.sh

# ROS side
python3 real_world/live/operator_loop.py --ros-args \
    -p instruction:="move toward the traffic cone" \
    -p artifact_dir:=real_world/live/artifacts
```

`serve_spot_ft.sh` exists because a few values silently ruin a run if typed wrong. It reads the
waypoint spacing, CCTV camera list and number of image slots from the checkpoint's `serving.json`,
refuses any override that contradicts them, derives `--step` from the head files present, checks GPU
headroom, and refuses a raw training checkpoint. The paper's Spot models were trained at a spacing of
0.19 m; `policy_server.py` alone defaults to 0.305 m.

The server and the operator loop talk over `tcp://127.0.0.1:5556` by default. Set `BIND` for the
server and the `server` parameter (or `OMNIVLA_SERVER`) for the loop when they run on different
machines.

### A LoRA fine-tune MUST be merged before serving

`inference/run_omnivla.py::define_model` loads the VLA with
`AutoModelForVision2Seq.from_pretrained(vla_path)` and then the two heads. **It never applies a LoRA
adapter.** Point it at a training checkpoint and you get base backbone weights driving fine-tuned
heads, with no error, just wrong numbers. `real_world/train/merge_lora_for_serving.py` merges the
adapter, asserts that the weights actually changed, and writes `serving.json`.

### Check the `transformers` fork

The checkpoints need the OpenVLA-OFT fork of `transformers`. The fork and stock 4.40.1 report the
same version; only the fork has the bidirectional attention the checkpoints were trained with. Check
the file, not the version:

```bash
python -c "import hashlib,pathlib,transformers.models.llama.modeling_llama as m; \
print(hashlib.md5(pathlib.Path(m.__file__).read_bytes()).hexdigest())"
# fork  = 752d33b65bc072540c1b7259ec18d2f9
# stock = f8588efe33303898815abda042b490c3
```

## Artifacts, one set per step

```
artifacts/step_0001/frontleft.png     decoded input
artifacts/step_0001/frontright.png    decoded input
artifacts/step_0001/onboard.png       stitched view actually fed to the model
artifacts/step_0001/behaviour.png     heading arrow + chunk, proposed waypoint starred
artifacts/step_0001/step.json         every number
artifacts/step_0001/send_goal.sh      the exact command, ready to run
```

## Topic names

Defaults, as published by the Spot ROS 2 driver:

| purpose | topic |
|---|---|
| left image | `/camera/frontleft/compressed` |
| right image | `/camera/frontright/compressed` |
| left intrinsics | `/camera/frontleft/camera_info` |
| right intrinsics | `/camera/frontright/camera_info` |
| extrinsics | `/tf_static` → `body` ← `frontleft_fisheye` / `frontright_fisheye` |
| action (printed, not called) | `/trajectory` |

**The compressed topic is `<base>/compressed`, not `<base>/image/compressed`.**
The driver publishes `image_topic_name + "/compressed"` and
`image_topic_name + "/image"` where `image_topic_name` is `camera/frontleft`.
Easy to get wrong; getting it wrong just means no frames ever arrive.

Override any of them on the command line if the graph changes:

```
-p left_topic:=/... -p right_topic:=/... -p left_info_topic:=/...
```

The two infrastructure cameras are read from `camera3` and `camera4` (parameter `cctv_cams`).

## Removing the human later

All delivery goes through the **Emitter seam** at the bottom of `operator_loop.py`.
`ManualEmitter` prints a command; `DirectEmitter` is the documented placeholder
for sending it over the action API. Enabling it requires **two independent
switches** — `ALLOW_DIRECT_CONTROL = True` in the file *and* `--allow-direct-control`
on the command line — plus `spot_msgs` on the path, which the read-only build
deliberately does not ship. Nothing else in the loop changes.

Before flipping it, revisit the abort story. `ManualEmitter` cannot stop the
robot because it never starts it. `DirectEmitter` breaks that property, so it
needs a real stop path: cancel the goal, spam zero `/cmd_vel` (the driver expires
velocity commands in 0.125 s, so one message is not enough), and call `/stop`.
