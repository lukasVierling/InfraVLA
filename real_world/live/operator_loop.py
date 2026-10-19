

from __future__ import annotations

import json
import math
import os
import pathlib
import pickle
import shutil
import subprocess
import datetime
import signal
import sys
import termios
import threading
import time
import tty

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import rclpy                              # noqa: E402
import zmq                                # noqa: E402
from rclpy.node import Node               # noqa: E402
from rclpy.qos import QoSPresetProfiles   # noqa: E402
from sensor_msgs.msg import CameraInfo, CompressedImage  # noqa: E402
from nav_msgs.msg import Odometry                        # noqa: E402
from std_msgs.msg import String                          # noqa: E402
import tf2_ros                            # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stitch_live import CAMS, FrontStitcher, quat_matrix  # noqa: E402

# =============================================================================
# HARD SAFETY INTERLOCK
# Leave False. Setting True only ARMS the direct path; --allow-direct-control
# is still required on top, and spot_msgs must be importable. Two independent
# switches so no single edit or typo can give the script control of the robot.
# =============================================================================
ALLOW_DIRECT_CONTROL = False

ABORT = threading.Event()
ABORT_KEYS = {" ", "q", "\x1b", "\x03"}   # space, q, ESC, Ctrl-C


def watch_keyboard() -> None:
    """Raw-mode stdin watcher. Stops the LOOP; cannot stop the robot."""
    if not sys.stdin.isatty():
        print("[abort] stdin is not a TTY — key abort disabled; Ctrl-C still works")
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not ABORT.is_set():
            if sys.stdin.read(1) in ABORT_KEYS:
                print("\r\n*** ABORT — loop stopping (robot NOT commanded by this script) ***\r")
                ABORT.set()
                return
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def resolve_instruction(object_name: str, pattern: str, verbatim: str) -> str:
    """Ask once for the navigation target and build the instruction from it.

    Precedence: an explicit `instruction` wins; otherwise `object_name` is
    slotted into `pattern`; otherwise we prompt. Runs BEFORE the keyboard
    watcher grabs the terminal, so ordinary line editing works here.
    """
    if verbatim.strip():
        return verbatim.strip()

    while True:
        name = object_name.strip()
        if not name:
            if not sys.stdin.isatty():
                raise SystemExit("no object_name and stdin is not a TTY — pass "
                                 "-p object_name:=... or -p instruction:=...")
            print("\nWhat should the robot navigate to?")
            print("  Give the bare object name, e.g.  traffic cone")
            print("  Common nouns ground far better than rare ones — 'traffic cone',")
            print("  'recycling bin' and 'fire extinguisher' were 17/17 on turn")
            print("  direction in the offline sweep; 'brick' pointed the wrong way.")
            try:
                name = input("  object > ").strip()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit("cancelled")
        if not name:
            continue
        # "the traffic cone" -> "traffic cone", so the pattern's article is not doubled
        low = name.lower()
        for article in ("the ", "a ", "an "):
            if low.startswith(article):
                name = name[len(article):].strip()
                break
        instruction = pattern.format(x=name)
        print(f'\n  instruction : "{instruction}"')
        print(f'  full prompt : "What action should the robot take to {instruction}?"')
        if not sys.stdin.isatty():
            return instruction
        try:
            ans = input("  Enter to accept, or type a different object > ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("cancelled")
        if not ans:
            return instruction
        object_name = ans          # retry with what they just typed


def install_signal_handlers() -> None:
    """SIGINT/SIGTERM set ABORT *and raise*.

    Setting a flag alone is not enough: under PEP 475 Python retries a syscall
    interrupted by a handler that returns normally, so a Ctrl-C at the per-step
    input() prompt printed the abort banner and then went straight back to
    waiting for a newline -- the loop looked hung and needed an extra Enter.
    Raising makes input() (and any interruptible wait) unwind into the existing
    `except (EOFError, KeyboardInterrupt)` and stop the loop immediately.
    """
    def handler(signum, frame):
        ABORT.set()
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


# =============================================================================
# ROS side — SUBSCRIPTIONS ONLY
# =============================================================================

class Perception(Node):
    """Reads images, intrinsics and tf. Creates no publisher or client, ever."""

    def __init__(self) -> None:
        super().__init__("omnivla_perception_readonly")
        self.declare_parameters("", [
            ("server", os.environ.get("OMNIVLA_SERVER", "tcp://127.0.0.1:5556")),
            # Navigation target. Normally left empty and typed in at startup;
            # the object name is slotted into `pattern` to build the instruction.
            ("object_name", ""),
            ("pattern", "move toward the {x}"),
            # Escape hatch: set this to bypass object_name/pattern entirely and
            # send a verbatim instruction (e.g. to test a non-canonical phrasing).
            ("instruction", ""),
            ("artifact_dir", "/artifacts"),
            # TOPIC NAMES as published by the Spot ROS 2 driver.
            # NOTE the compressed topic is "<base>/compressed", NOT
            # "<base>/image/compressed": the driver builds it as
            # image_topic_name + "/compressed" where image_topic_name is
            # "camera/frontleft", while the raw one is image_topic_name + "/image".
            ("left_topic", "/camera/frontleft/compressed"),
            ("right_topic", "/camera/frontright/compressed"),
            ("left_info_topic", "/camera/frontleft/camera_info"),
            ("right_info_topic", "/camera/frontright/camera_info"),
            # External CCTV cameras for the CCTV checkpoints. Comma-separated,
            # in TRAINING slot order; empty string disables. NOTE the externals
            # publish "<cam>/image_raw/compressed", unlike Spot's "<base>/compressed".
            ("cctv_cams", "camera3,camera4"),
            # REFERENCE-FRAME DIAGNOSTIC (measurement, never an intervention).
            # A canonical spawn view saved once per evaluation run. On step 1 the
            # loop runs a second inference with this ego frame in place of the
            # live one (live CCTV kept). The robot is always driven by the live
            # prediction; the reference prediction is only logged, to quantify how
            # much this trial's spawn imprecision moved the aim.
            ("reference_frame", ""),        # path to main_frame.png; "" = disabled
            ("save_reference_frame", False),  # True: write this run's step-1 view
                                              # to <artifact_dir>/main_frame.png
            ("cctv_max_age_s", 1.0),      # older frames are treated as missing
            ("body_frame", "body"),
            ("left_frame", "frontleft_fisheye"),
            ("right_frame", "frontright_fisheye"),
            # what the emitted command will target
            ("trajectory_action", "/trajectory"),
            # geometry / model
            ("spacing_m", 0.305),
            ("waypoint_index", 4),        # "the 5th waypoint", 0-based
            ("hfov_deg", 110.0),
            ("plane_distance_m", 2.0),
            # goal shaping
            ("duration_s", 8),
            # Kinematic feasibility clamp. wp[k] is (k+1) ticks ahead (verified:
            # mean |xy| / (k+1) is flat at 0.12 m across all 8 waypoints), so the
            # furthest the robot could HAVE travelled to reach it is
            #     max_speed_mps * (waypoint_index + 1) / tick_hz.
            # Beyond that the prediction is not a long stride, it is claiming the
            # robot teleported -- which is the only thing a clamp should catch.
            # A fixed cap would not scale with the spacing and can silently
            # truncate ground-truth-sized steps.
            ("max_speed_mps", 0.578),     # measured max over the recorded demonstrations
            ("tick_hz", 3.0),             # chunk waypoints are 1 tick apart
            ("max_step_m", 0.0),          # >0 overrides the derived cap; 0 = derive
            ("precise_positioning", False),
            ("disable_obstacle_avoidance", False),
            # ---- recording -------------------------------------------------
            # Every launch gets its OWN timestamped directory under run_root, so
            # a restart can no longer overwrite the previous run at step_0001.
            ("run_root", ""),             # default: <artifact_dir>/runs
            ("record_bag", True),         # rosbag2 per run -> single source of truth
            ("bag_topics", "/camera/frontleft/compressed,/camera/frontright/compressed,"
                           "/camera/frontleft/camera_info,/camera/frontright/camera_info,"
                           "/camera3/image_raw/compressed,/camera4/image_raw/compressed,"
                           "/odometry,/odometry/twist,/tf,/tf_static,/omnivla/step,"
                           "/trajectory/_action/feedback,/trajectory/_action/status"),
            # The goal you send is an action SERVICE call, invisible to a normal
            # topic recording. Recording it makes the bag reconstruct the commands
            # too, not just their effects.
            ("bag_services", "/trajectory/_action/send_goal,/trajectory/_action/get_result"),
            ("odom_topic", "/odometry"),
        ])
        self.p = lambda k: self.get_parameter(k).value  # noqa: E731

        # One directory per launch: <run_root>/<UTCstamp>_<arm>_<slug>/
        root = self.p("run_root") or str(pathlib.Path(self.p("artifact_dir")) / "runs")
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        arm = "cctv" if self.cctv_cams_raw() else "egoonly"
        self.out_root = pathlib.Path(root) / f"{stamp}_{arm}"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.bag_dir = self.out_root / "bag"
        self._bag_proc = None

        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._info: dict[str, np.ndarray] = {}
        self._src_hw: dict[str, tuple] = {}

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(CompressedImage, self.p("left_topic"),
                                 lambda m: self._on_img("frontleft", m), qos)
        self.create_subscription(CompressedImage, self.p("right_topic"),
                                 lambda m: self._on_img("frontright", m), qos)
        self.create_subscription(CameraInfo, self.p("left_info_topic"),
                                 lambda m: self._on_info("frontleft", m), 10)
        self.create_subscription(CameraInfo, self.p("right_info_topic"),
                                 lambda m: self._on_info("frontright", m), 10)
        # CCTV: keep the RAW jpeg bytes (the server decodes); stamp with local
        # receipt time — header stamps live in the mags' clock domain.
        self.cctv_cams = [c.strip() for c in str(self.p("cctv_cams")).split(",") if c.strip()]
        self._cctv: dict[str, tuple[bytes, float]] = {}
        for cam in self.cctv_cams:
            self.create_subscription(
                CompressedImage, f"/{cam}/image_raw/compressed",
                lambda m, c=cam: self._on_cctv(c, m), qos)
        # Robot pose per step. Without this a run records what was COMMANDED but
        # never where the robot actually was, so distance-to-object cannot be
        # computed afterwards -- the whole point of the record.
        self._odom = None
        self.create_subscription(Odometry, self.p("odom_topic"), self._on_odom, 10)
        # Diagnostic publisher (NOT a command topic): one JSON per step so the
        # rosbag captures the model side -- chunk, instruction, goal, checkpoint.
        # The read-only guarantee is "cannot command the robot"; this publishes to
        # /omnivla/step, which nothing actuates, and there is still no action
        # client and no publisher on any command topic.
        self.step_pub = self.create_publisher(String, "/omnivla/step", 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.stitcher: FrontStitcher | None = None
        self.sock = zmq.Context().socket(zmq.REQ)
        # 15 s, not 120: a wrong --server port should fail fast with
        # zmq.error.Again rather than look like a long hang.
        self.sock.setsockopt(zmq.RCVTIMEO, 15000)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.p("server"))

    def cctv_cams_raw(self):
        return [c.strip() for c in str(self.p("cctv_cams")).split(",") if c.strip()]

    def pose(self):
        """Latest odometry as a plain dict, or None if nothing has arrived."""
        o = self._odom
        if o is None:
            return None
        p_, q = o.pose.pose.position, o.pose.pose.orientation
        return {"x": round(p_.x, 4), "y": round(p_.y, 4),
                "yaw_deg": round(math.degrees(math.atan2(
                    2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))), 2),
                "frame": o.header.frame_id,
                "stamp": o.header.stamp.sec + o.header.stamp.nanosec * 1e-9}

    def server_info(self):
        """Ask the server which checkpoint it is serving. Never fatal."""
        try:
            self.sock.send(pickle.dumps({"cmd": "info"}))
            r = pickle.loads(self.sock.recv())
            return r if r.get("ok") else {"error": r.get("error")}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def start_bag(self):
        if not bool(self.p("record_bag")):
            print("  (bag recording disabled)")
            return
        topics = [t for t in str(self.p("bag_topics")).split(",") if t.strip()]
        svcs = [t for t in str(self.p("bag_services")).split(",") if t.strip()]
        cmd = ["ros2", "bag", "record", "-o", str(self.bag_dir), *topics]
        if svcs:
            cmd += ["--services", *svcs]
        try:
            # stdin=DEVNULL is load-bearing. rosbag2 reads the keyboard for its
            # pause/resume controls, so an inherited TTY means the recorder and
            # the per-step prompt fight over every keystroke: Enter lands in the
            # recorder and input() sees EOF, which aborted the loop after step 1.
            # Same failure as the old keyboard-watcher thread, via a subprocess.
            self._bag_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.STDOUT,
                                              start_new_session=True)
            print(f"  recording bag -> {self.bag_dir}")
        except Exception as exc:
            print(f"  WARNING: could not start rosbag ({exc}); continuing without it")

    def stop_bag(self):
        """SIGINT, not kill: rosbag2 only finalises metadata.yaml on a clean stop."""
        if self._bag_proc is None or self._bag_proc.poll() is not None:
            return
        self._bag_proc.send_signal(signal.SIGINT)
        try:
            self._bag_proc.wait(timeout=20)
            print("  bag closed cleanly")
        except subprocess.TimeoutExpired:
            self._bag_proc.kill()
            print("  WARNING: bag did not close in 20 s; killed (metadata may be incomplete)")

    # ---- callbacks --------------------------------------------------------
    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_img(self, cam: str, msg: CompressedImage) -> None:
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        with self._lock:
            self._frames[cam] = bgr[:, :, ::-1].copy()   # -> RGB
            self._src_hw[cam] = bgr.shape[:2]

    def _on_info(self, cam: str, msg: CameraInfo) -> None:
        with self._lock:
            self._info[cam] = np.array(msg.k).reshape(3, 3)

    def _on_cctv(self, cam: str, msg: CompressedImage) -> None:
        with self._lock:
            self._cctv[cam] = (bytes(msg.data), time.time())

    # ---- setup ------------------------------------------------------------
    def wait_for_inputs(self, timeout: float = 30.0) -> bool:
        print("waiting for both cameras, camera_info and tf ...")
        t0 = time.time()
        while time.time() - t0 < timeout and not ABORT.is_set():
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                ready = len(self._frames) == 2 and len(self._info) == 2
            if ready and self._build_stitcher():
                return True
        with self._lock:
            print(f"  got frames from {sorted(self._frames)}, info from {sorted(self._info)}")
        return False

    def _build_stitcher(self) -> bool:
        try:
            body_T_cam = {}
            for cam, frame in (("frontleft", self.p("left_frame")),
                               ("frontright", self.p("right_frame"))):
                tr = self.tf_buffer.lookup_transform(self.p("body_frame"), frame,
                                                     rclpy.time.Time())
                q, t = tr.transform.rotation, tr.transform.translation
                body_T_cam[cam] = quat_matrix((q.x, q.y, q.z, q.w), (t.x, t.y, t.z))
        except Exception:
            return False
        with self._lock:
            K, hw = dict(self._info), dict(self._src_hw)
        self.stitcher = FrontStitcher(body_T_cam, K, hw,
                                      distance_m=float(self.p("plane_distance_m")),
                                      hfov_deg=float(self.p("hfov_deg")))
        print(f"stitcher ready: {self.stitcher.out_size[0]}x{self.stitcher.out_size[1]}, "
              f"hfov {self.stitcher.hfov:.0f} deg, vfov {self.stitcher.vfov:.0f} deg, "
              f"f={self.stitcher.f:.1f} px")
        return True

    # ---- per-step ---------------------------------------------------------
    def capture(self):
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.05)
        with self._lock:
            if len(self._frames) < 2:
                return None, None
            frames = {c: self._frames[c].copy() for c in CAMS}
        return frames, self.stitcher.composite(frames)

    def set_instruction(self, instruction: str) -> None:
        """Arm the server in language-only mode. MUST precede any step().

        The server keeps the goal on its side (it sets the run_omnivla module
        globals that select the modality, and builds the Inference object), so a
        step() before this raises on the server. Sent once at startup; call again
        if the instruction ever changes mid-session.
        """
        self.sock.send(pickle.dumps({
            "cmd": "reset",
            "goal": {"mode": "language", "instruction": instruction},
        }))
        try:
            rep = pickle.loads(self.sock.recv())
        except zmq.error.Again:
            # print, then exit with an int: main() does int(exc.code or 0).
            print(f"\nERROR: no reply from the policy server at {self.p('server')} "
                  f"within 15 s.\n"
                  f"       Is it running, and is the port right?  ss -ltn | grep 55\n"
                  f"       then relaunch with -p server:=tcp://127.0.0.1:<port>",
                  file=sys.stderr)
            raise SystemExit(3)
        if not rep.get("ok"):
            raise RuntimeError(f"policy server refused the instruction: {rep.get('error')}")
        print(f'server armed, language-only mode')
        print(f'  prompt sent to the LLM: '
              f'"What action should the robot take to {instruction}?"')

    def query_model(self, onboard: np.ndarray, record: bool = True) -> dict:
        ok, jpg = cv2.imencode(".jpg", onboard[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        req = {"cmd": "step", "image": jpg.tobytes(),
               "x": 0.0, "y": 0.0, "yaw": 0.0}
        # Kept so save_artifacts can write the EXACT frames the model was given.
        # Without this a step is unauditable: you can see the ego view but not
        # the external views, which is half the model's input on a CCTV arm.
        # record=False (reference query) must not clobber the live step's record.
        prev_cctv = getattr(self, "_last_cctv", {})
        prev_missing = getattr(self, "_last_cctv_missing", [])
        self._last_cctv: dict[str, bytes] = {}
        self._last_cctv_missing: list[str] = []
        if self.cctv_cams:
            now = time.time()
            max_age = float(self.p("cctv_max_age_s"))
            cctv: dict[str, bytes] = {}
            with self._lock:
                snap = dict(self._cctv)
            for cam in self.cctv_cams:
                raw = snap.get(cam)
                if raw is not None and now - raw[1] <= max_age:
                    cctv[cam] = raw[0]
            missing = [c for c in self.cctv_cams if c not in cctv]
            if missing:
                # The server zero-fills + masks absent slots (training's dropped-view
                # path), so this degrades rather than fails — but say so loudly:
                # an unnoticed dead camera silently tests a weaker policy.
                print(f"  !! CCTV missing/stale: {missing} — serving degraded")
            print(f"  cctv attached: {sorted(cctv)}")
            req["cctv"] = cctv
            self._last_cctv, self._last_cctv_missing = cctv, missing
        if not record:                      # restore the live step's audit trail
            self._last_cctv, self._last_cctv_missing = prev_cctv, prev_missing
        self.sock.send(pickle.dumps(req))
        rep = pickle.loads(self.sock.recv())
        if not rep.get("ok"):
            raise RuntimeError(f"policy server: {rep.get('error')}")
        return rep

    def build_goal(self, chunk: np.ndarray, spacing: float) -> dict:
        """Chunk row -> the goal FIELDS. Pure data; nothing is sent here.

        Preprocessing, in order:
          0. chunk[N] = [x, y, cos, sin], normalised units, body frame (+x fwd, +y left)
          1. x,y * spacing            -> metres     (cos/sin are a direction, not scaled)
          2. yaw = atan2(sin, cos)    -> quaternion about Z
          3. frame check: OmniVLA (+x fwd, +y left) == Spot body -> no sign flip
          4. clamp to the kinematic cap, max_speed_mps * (idx+1) / tick_hz,
             so one approved step can never exceed what the robot could
             physically have walked in the time that waypoint represents
        """
        idx = int(self.p("waypoint_index"))
        wp = chunk[idx]
        fwd, left = float(wp[0]) * spacing, float(wp[1]) * spacing
        yaw = math.atan2(float(wp[3]), float(wp[2]))
        dist = math.hypot(fwd, left)

        clamped = False
        cap = float(self.p("max_step_m"))
        cap_source = "max_step_m override"
        if cap <= 0.0:
            ticks = idx + 1
            cap = float(self.p("max_speed_mps")) * ticks / float(self.p("tick_hz"))
            cap_source = (f'{self.p("max_speed_mps")} m/s * {ticks} ticks '
                          f'/ {self.p("tick_hz")} Hz')
        if dist > cap:
            fwd, left = fwd * cap / dist, left * cap / dist
            clamped, dist = True, cap

        return {
            "action": self.p("trajectory_action"),
            "frame_id": self.p("body_frame"),
            "waypoint_index": idx,
            "waypoint_normalised": [round(float(v), 5) for v in wp],
            "spacing_m": spacing,
            "forward_m": round(fwd, 4),
            "left_m": round(left, 4),
            "distance_m": round(dist, 4),
            "bearing_deg": round(math.degrees(math.atan2(left, fwd)), 2),
            "yaw_deg": round(math.degrees(yaw), 2),
            "qz": round(math.sin(yaw / 2), 7),
            "qw": round(math.cos(yaw / 2), 7),
            "clamped_to_max_step": clamped,
            "max_step_cap_m": round(cap, 4),
            "max_step_cap_source": cap_source,
            "duration_s": int(self.p("duration_s")),
            "precise_positioning": bool(self.p("precise_positioning")),
            "disable_obstacle_avoidance": bool(self.p("disable_obstacle_avoidance")),
        }


# =============================================================================
# Artifacts
# =============================================================================

def step_record(step_idx, instruction, g, cctv, cctv_missing, pose, chunk_full,
                ref_diag=None):
    """The complete per-step record. Written to step.json AND published on
    /omnivla/step, so the rosbag carries the model side of the loop too."""
    return {
        "step": step_idx,
        "instruction": instruction,
        "t_wall": time.time(),
        "t_iso": datetime.datetime.now().isoformat(timespec="seconds"),
        "pose_odom": pose,                    # where the robot ACTUALLY was
        "chunk_full": (None if chunk_full is None
                       else [[round(float(v), 5) for v in row] for row in chunk_full]),
        "cctv_attached": sorted(cctv) if cctv else [],
        "cctv_missing": sorted(cctv_missing) if cctv_missing else [],
        # present on step 1 only, when a reference frame was configured:
        # the SAME model re-queried with the canonical spawn view (live CCTV kept).
        # Diagnostic only - never used to command the robot.
        "reference_diagnostic": ref_diag,
        **g}


def save_artifacts(d: pathlib.Path, frames, onboard, chunk_m, g,
                   instruction, stitcher, command: str,
                   cctv: dict | None = None, cctv_missing: list | None = None,
                   pose: dict | None = None, chunk_full=None, step_idx: int = 0,
                   ref_diag: dict | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for cam in CAMS:
        cv2.imwrite(str(d / f"{cam}.png"), frames[cam][:, :, ::-1])
    cv2.imwrite(str(d / "onboard.png"), onboard[:, :, ::-1])
    # The external views AS SENT -- raw bytes straight from the topic, not
    # re-encoded, so what lands on disk is byte-identical to the model's input.
    if cctv:
        (d / "cctv").mkdir(exist_ok=True)
        for cam, raw in sorted(cctv.items()):
            (d / "cctv" / f"{cam}.jpg").write_bytes(raw)

    H, W = onboard.shape[:2]
    fig, (ax_i, ax_w) = plt.subplots(1, 2, figsize=(15, 6),
                                     gridspec_kw={"width_ratios": [1.35, 1]})
    ax_i.imshow(onboard)
    ax_i.axvline(W / 2, color="white", lw=1, ls=":", alpha=0.7)
    col = stitcher.project(g["bearing_deg"])
    ax_i.annotate("", xy=(col, H * 0.32), xytext=(W / 2, H * 0.97),
                  arrowprops=dict(color="#ff3b30", width=4, headwidth=15, alpha=0.9))
    ax_i.text(col, H * 0.26, f'{g["bearing_deg"]:+.1f}°', color="#ff3b30",
              fontsize=14, fontweight="bold", ha="center",
              bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
    ax_i.set_xlim(0, W); ax_i.set_ylim(H, 0); ax_i.axis("off")
    ax_i.set_title("onboard (stitched) — red = heading of the proposed step", fontsize=11)

    ax_w.plot(np.insert(-chunk_m[:, 1], 0, 0.0), np.insert(chunk_m[:, 0], 0, 0.0),
              "o-", lw=3, ms=9, color="#0a84ff", label="full chunk")
    ax_w.plot(-g["left_m"], g["forward_m"], "*", ms=22, color="#ff3b30",
              label=f'proposed: wp[{g["waypoint_index"]}]')
    ax_w.plot(0, 0, "s", ms=11, color="#1c1c1e")
    lim = max(0.6, float(np.abs(chunk_m[:, 1]).max()) * 2.2)
    ax_w.set_xlim(-lim, lim)
    ax_w.set_ylim(-0.05, max(1.2, float(chunk_m[:, 0].max()) * 1.25))
    ax_w.set_xlabel("←  left      rightward displacement [m]      right  →")
    ax_w.set_ylabel("forward [m]")
    ax_w.grid(alpha=0.3); ax_w.legend(loc="upper right")
    ax_w.set_title("chunk (blue) and the ONE waypoint proposed (red)", fontsize=11)
    fig.suptitle(f'"What action should the robot take to {instruction}?"',
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(d / "behaviour.png", dpi=85)
    plt.close(fig)

    (d / "step.json").write_text(json.dumps(step_record(
        step_idx, instruction, g, cctv, cctv_missing, pose, chunk_full, ref_diag), indent=2))
    sh = d / "send_goal.sh"
    sh.write_text("#!/usr/bin/env bash\n# Generated — review before running.\n"
                  f"# instruction: {instruction}\n{command}\n")
    sh.chmod(0o755)


# =============================================================================
# Emitter seam — the ONLY place delivery happens.
# Swap ManualEmitter for DirectEmitter to remove the human. Nothing else changes.
# =============================================================================

def format_send_goal(g: dict) -> str:
    """Build the exact `ros2 action send_goal` command for this step."""
    payload = (
        "{"
        f"target_pose: {{header: {{frame_id: '{g['frame_id']}'}}, "
        f"pose: {{position: {{x: {g['forward_m']}, y: {g['left_m']}, z: 0.0}}, "
        f"orientation: {{x: 0.0, y: 0.0, z: {g['qz']}, w: {g['qw']}}}}}}}, "
        f"duration: {{sec: {g['duration_s']}, nanosec: 0}}, "
        f"precise_positioning: {str(g['precise_positioning']).lower()}, "
        f"disable_obstacle_avoidance: {str(g['disable_obstacle_avoidance']).lower()}"
        "}"
    )
    return (f"ros2 action send_goal --feedback {g['action']} "
            f"spot_msgs/action/Trajectory \"{payload}\"")


class ManualEmitter:
    """Prints the command. A human runs it. The script stays read-only."""

    can_command_robot = False

    def deliver(self, g: dict, step_dir: pathlib.Path, command: str) -> None:
        print("\n" + "=" * 78)
        print(f"  REVIEW   {step_dir}")
        print("    onboard.png   behaviour.png   step.json   send_goal.sh")
        print("-" * 78)
        print(f"  forward   {g['forward_m']:+.3f} m")
        print(f"  left      {g['left_m']:+.3f} m")
        print(f"  distance  {g['distance_m']:.3f} m"
              + (f'   [CLAMPED to {g["max_step_cap_m"]} m]'
                 if g["clamped_to_max_step"] else ""))
        print(f"  bearing   {g['bearing_deg']:+.1f} deg")
        print(f"  yaw       {g['yaw_deg']:+.1f} deg")
        print(f"  timeout   {g['duration_s']} s     frame '{g['frame_id']}'")
        print(f"  obstacle avoidance: {'OFF' if g['disable_obstacle_avoidance'] else 'ON'}")
        print("-" * 78)
        print("  RUN THIS YOURSELF (robot side, container with spot_msgs):\n")
        print(command)
        print("\n" + "=" * 78)
        print("  Press Enter when you are done (or have chosen not to run it)")
        print("  to compute the next step.  SPACE / q / Ctrl-C aborts the loop.")


class DirectEmitter:
    """Autonomous delivery. NOT REACHABLE — see ALLOW_DIRECT_CONTROL.

    Left in place as the documented seam for later. To enable, you must:
      1. set ALLOW_DIRECT_CONTROL = True at the top of this file,
      2. pass --allow-direct-control,
      3. have spot_msgs importable (the read-only path does not need it).
    It also means the script gains publishers/clients, so revisit the abort
    story first: ManualEmitter cannot stop the robot because it never starts it,
    which stops being true here.
    """

    can_command_robot = True

    def __init__(self, node) -> None:
        raise NotImplementedError(
            "Direct control is disabled. This script is read-only toward the robot.")


def make_emitter(node, allow_flag: bool):
    if allow_flag and ALLOW_DIRECT_CONTROL:
        return DirectEmitter(node)
    return ManualEmitter()


# =============================================================================

def finalise_run(run_dir: pathlib.Path) -> None:
    """Keep or drop the run, and give it a name. Runs AFTER the bag is closed.

    Nothing is deleted without an explicit typed 'discard' -- a stray keypress at
    the end of a rollout must not destroy the recording.
    """
    if not run_dir.exists():
        return
    steps = sorted(run_dir.glob("step_*"))
    size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file()) / 1e6
    print(f"\n{'=' * 66}\n  RUN: {run_dir.name}   {len(steps)} steps, {size:.0f} MB")
    if not sys.stdin.isatty():
        print(f"  (not a TTY — kept as-is)\n{'=' * 66}")
        return
    print("  [Enter] keep    [name] keep and rename    [discard] delete it")
    try:
        ans = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans.lower() == "discard":
        confirm = input(f"  type the run name to confirm deletion ({run_dir.name}): ").strip()
        if confirm == run_dir.name:
            shutil.rmtree(run_dir, ignore_errors=True)
            print("  discarded.")
        else:
            print("  name did not match — KEPT.")
        print("=" * 66)
        return

    final = run_dir
    if ans:
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in ans)
        dest = run_dir.parent / f"{run_dir.name}_{safe}"
        try:
            run_dir.rename(dest); final = dest
        except OSError as exc:
            print(f"  rename failed ({exc}); kept as {run_dir}")

    # Outcome goes into run.json, so scene, result and recording live together
    # instead of in a spreadsheet that has to be joined by hand afterwards.
    def ask(prompt, default=""):
        try:
            return input(prompt).strip() or default
        except (EOFError, KeyboardInterrupt):
            return default
    scene = ask("  scene id (from the sampler CSV, e.g. S03; blank = none) > ")
    succ = ask("  success? [y/N] > ").lower().startswith("y")
    note = ask("  note > ")
    meta_path = final / "run.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["result"] = {
            "scene_id": scene or None,
            "success": succ,
            "note": note,
            "steps": len(sorted(final.glob("step_*"))),
            "ended_iso": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  recorded: scene={scene or '-'}  success={succ}  -> {meta_path}")
    except Exception as exc:
        print(f"  WARNING: could not write the result into run.json: {exc}")
    print(f"  kept as {final}")
    print("=" * 66)


def main() -> None:
    allow_direct = "--allow-direct-control" in sys.argv
    rclpy.init(args=[a for a in sys.argv if a != "--allow-direct-control"])
    # AFTER rclpy.init on purpose: rclpy installs its own SIGINT handler during
    # init, which would overwrite ours and swallow Ctrl-C at the prompt (it only
    # flags its internal shutdown; input() keeps waiting for a newline).
    install_signal_handlers()
    node = Perception()
    code = 0
    try:
        emitter = make_emitter(node, allow_direct)
        if emitter.can_command_robot:
            print("!!! DIRECT CONTROL ARMED !!!")
        else:
            print("read-only mode: this script cannot command or stop the robot.")

        if not node.wait_for_inputs():
            print("\nERROR: cameras / camera_info / tf did not all arrive within 30 s.")
            print("       Most likely the topic names differ. Send me the published")
            print("       list and I will pin them; meanwhile override with e.g.")
            print("       -p left_topic:=/... -p right_topic:=/...")
            raise SystemExit(2)

        # Asked once, up front, then fixed for the whole session.
        instruction = resolve_instruction(node.p("object_name"),
                                          node.p("pattern"),
                                          node.p("instruction"))
        node.set_instruction(instruction)      # must precede the first step()

        info = node.server_info()
        run_meta = {
            "started_iso": datetime.datetime.now().isoformat(timespec="seconds"),
            "instruction": instruction,
            "server": node.p("server"),
            "model": info,                     # checkpoint, step, spacing, cams
            "params": {k: node.p(k) for k in (
                "cctv_cams", "spacing_m", "waypoint_index", "max_step_m",
                "max_speed_mps", "tick_hz", "duration_s", "trajectory_action",
                "precise_positioning", "disable_obstacle_avoidance",
                "record_bag", "bag_topics", "bag_services")},
            "host": os.uname().nodename,
        }
        (node.out_root / "run.json").write_text(json.dumps(run_meta, indent=2))
        print(f"\nrun dir : {node.out_root}")
        if info.get("vla_path"):
            print(f"model   : {pathlib.Path(info['vla_path']).name}  step {info.get('step')}"
                  f"  cams={info.get('cctv_cams') or 'none'}  spacing {info.get('spacing_m')}")
        else:
            print(f"model   : UNKNOWN ({info.get('error')}) — server predates the info command")
        node.start_bag()

        # NOTE: no keyboard-watcher thread. It read stdin in cbreak mode while
        # the per-step prompt below read the same fd, so the two raced for every
        # keystroke and the abort key landed in whichever won. The prompt is now
        # the only reader of stdin.
        # Prefer the SERVER's spacing: the local param default is 0.305 and
        # printing that was misleading on a 0.19 checkpoint.
        spacing = float(info.get("spacing_m") or node.p("spacing_m"))
        print(f'\nspacing {spacing} m/unit'
              + ("" if info.get("spacing_m") else "   (local default — server did not report)"))
        print("at the prompt: Enter = next step, q = quit. Ctrl-C also quits.\n")

        step = 0
        while rclpy.ok() and not ABORT.is_set():
            step += 1
            step_dir = node.out_root / f"step_{step:04d}"
            print(f"\n--- step {step}: capture ---")
            frames, onboard = node.capture()
            if onboard is None:
                print("  no frames yet; retrying")
                continue
            print("--- inference ---")
            rep = node.query_model(onboard)
            chunk = np.array(rep["waypoints"])
            spacing = float(rep.get("spacing_m", spacing))

            # ---- step-1 reference diagnostic -----------------------------
            ref_diag = None
            if step == 1:
                if node.p("save_reference_frame"):
                    rp = pathlib.Path(node.p("artifact_dir")) / "main_frame.png"
                    cv2.imwrite(str(rp), onboard[:, :, ::-1])
                    print(f"  reference frame WRITTEN -> {rp}")
                ref_path = str(node.p("reference_frame") or "")
                if ref_path and pathlib.Path(ref_path).is_file():
                    ref_bgr = cv2.imread(ref_path)
                    if ref_bgr is None:
                        print(f"  !! reference frame unreadable: {ref_path}")
                    elif ref_bgr.shape[:2] != onboard.shape[:2]:
                        print(f"  !! reference frame is {ref_bgr.shape[1]}x{ref_bgr.shape[0]}, "
                              f"live is {onboard.shape[1]}x{onboard.shape[0]} - skipped")
                    else:
                        rrep = node.query_model(ref_bgr[:, :, ::-1], record=False)
                        rchunk = np.array(rrep["waypoints"])
                        d_end = float(np.linalg.norm(
                            (rchunk[-1, :2] - chunk[-1, :2]) * spacing))
                        b_live = math.degrees(math.atan2(chunk[-1, 1], chunk[-1, 0]))
                        b_ref = math.degrees(math.atan2(rchunk[-1, 1], rchunk[-1, 0]))
                        ref_diag = {
                            "reference_frame": ref_path,
                            "chunk_full": rchunk.tolist(),
                            "chunk_end_delta_m": round(d_end, 4),
                            "bearing_live_deg": round(b_live, 2),
                            "bearing_reference_deg": round(b_ref, 2),
                            "bearing_delta_deg": round(b_live - b_ref, 2),
                        }
                        print(f"  REFERENCE DIAGNOSTIC  aim delta {d_end:.3f} m   "
                              f"bearing live {b_live:+.1f} vs ref {b_ref:+.1f} "
                              f"(delta {b_live - b_ref:+.1f} deg)")
                        print("     (logged only - the robot follows the LIVE prediction)")
                elif ref_path:
                    print(f"  !! reference_frame not found: {ref_path}")
                node._ref_diag = ref_diag
            g = node.build_goal(chunk, spacing)
            command = format_send_goal(g)
            pose = node.pose()
            save_artifacts(step_dir, frames, onboard, chunk[:, :2] * spacing,
                           g, instruction, node.stitcher, command,
                           cctv=getattr(node, "_last_cctv", None),
                           cctv_missing=getattr(node, "_last_cctv_missing", None),
                           pose=pose, chunk_full=chunk, step_idx=step,
                           ref_diag=getattr(node, "_ref_diag", None))
            # same record onto the bag's timeline
            node.step_pub.publish(String(data=json.dumps(step_record(
                step, instruction, g, getattr(node, "_last_cctv", None),
                getattr(node, "_last_cctv_missing", None), pose, chunk,
                getattr(node, "_ref_diag", None)))))
            if pose is None:
                print("  !! no odometry yet — pose not recorded for this step")
            if ABORT.is_set():
                break
            emitter.deliver(g, step_dir, command)
            try:
                ans = input("  > ").strip().lower()
                if ans in ("q", "quit", "a", "abort", "stop"):
                    print("  quitting.")
                    ABORT.set()
                    break
                # live waypoint change: type "wp N" or just "N" (0-7);
                # takes effect from the NEXT inference (this step already sent).
                m_wp = ans.split()
                if (len(m_wp) == 2 and m_wp[0] == "wp" and m_wp[1].isdigit()) or \
                   (len(m_wp) == 1 and ans.isdigit()):
                    new_idx = int(m_wp[-1])
                    if 0 <= new_idx <= 7:
                        from rclpy.parameter import Parameter as _P
                        node.set_parameters([_P("waypoint_index", value=new_idx)])
                        print(f"  waypoint_index -> {new_idx} "
                              f"(~{(new_idx + 1) * 0.19 * 0.7:.2f} m/step) from next step")
                    else:
                        print("  waypoint index must be 0-7")
            except (EOFError, KeyboardInterrupt):
                print("\n*** ABORT — loop stopping (robot NOT commanded by this script) ***")
                ABORT.set()
                break
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        try:
            node.stop_bag()
        except Exception as exc:
            print(f"  WARNING: bag teardown failed: {exc}")
        run_dir = node.out_root
        node.destroy_node()
        rclpy.shutdown()
        print("\nloop stopped. (No command was ever sent to the robot by this script.)")
        finalise_run(run_dir)
    sys.exit(code)


if __name__ == "__main__":
    main()
