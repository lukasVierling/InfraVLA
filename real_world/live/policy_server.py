
from __future__ import annotations

import argparse
import io
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import zmq
from PIL import Image

def _find_repo_root(start: str) -> str:
    """Walk up until we find the InfraVLA checkout.

    Located by marker rather than by a fixed number of `..` hops: this file has
    already moved once (realrobot/ -> real_world/live/) and a hardcoded depth
    silently resolved to the wrong directory, giving an unhelpful
    "No module named 'run_omnivla'".
    """
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, "inference", "run_omnivla.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit(
                "could not locate the InfraVLA repo root above "
                f"{os.path.abspath(start)} (looked for inference/run_omnivla.py). "
                "Set INFRAVLA_ROOT to the checkout path.")
        d = parent


_REPO_ROOT = os.environ.get("INFRAVLA_ROOT") or _find_repo_root(os.path.dirname(__file__))
for _p in (_REPO_ROOT, f"{_REPO_ROOT}/inference", f"{_REPO_ROOT}/isaac_sim"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_omnivla as ro  # noqa: E402
from run_omnivla import Inference, InferenceConfig, define_model  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402


class OnboardOnlyPolicy:
    """Ego-camera OmniVLA: (image, pose) -> waypoint chunk -> (v, w) twist.

    The goal-pose encoding and the twist controller are lifted from
    isaac_sim/policies/policy.py (`_forward` / `__call__`) rather than from
    run_omnivla.py's GPS branch: the sim path is the one that has actually been
    validated closed-loop against these checkpoints, and it uses a plain
    (forward, left) robot frame instead of the UTM/compass swap.
    """

    def __init__(self, vla_path: str, resume_step: int, tick_hz: float,
                 max_v: float, max_w: float, waypoint_select: int,
                 spacing_m: float, thres_dist_m: float = 30.0):
        self.dt_s = 1.0 / tick_hz
        self.max_v, self.max_w = max_v, max_w
        self.waypoint_select = waypoint_select
        self.thres_dist_m = thres_dist_m
        # DENORMALISATION SCALE — deliberately NOT derived from max_v.
        # OmniVLA emits waypoints in units of "one waypoint spacing"; the metric
        # value of that unit is a property of the training mixture, not of the
        # speed we happen to cap the robot at. Conflating the two is easy (both
        # look like max_v/control_hz) and silently rescales every distance.
        # The GNM mixture's per-dataset constants are cory_hall 0.06,
        # go_stanford 0.12, recon 0.25, sacson 0.255, seattle 0.35, scand 0.38,
        # tartan_drive 0.72 -> mean 0.305. The model cannot know which embodiment
        # it is on, so the mixture mean is the defensible default.
        self.waypoint_spacing = spacing_m

        cfg = InferenceConfig(resume=True, vla_path=vla_path, resume_step=resume_step,
                              num_images_in_input=2)
        (self.vla, self.action_head, self.pose_projector, self.device_id,
         self.num_patches, self.action_tokenizer, self.processor) = define_model(cfg)
        self.inference = None
        self.goal = None
        self.goal_image = None

    def reset(self, goal: dict) -> None:
        mode = goal["mode"]
        # run_omnivla.run_forward_pass reads these as MODULE globals.
        ro.satellite = False
        ro.pose_goal = mode == "pose"
        ro.image_goal = mode == "image"
        ro.lan_prompt = mode == "language"

        self.goal = goal
        self.goal_image = (Image.open(goal["path"]).convert("RGB")
                           if mode == "image" else None)
        self.inference = Inference(
            save_dir=goal.get("save_dir", "/tmp/omnivla_realrobot"),
            lan_inst_prompt=goal.get("instruction", "xxxx"),
            goal_image_PIL=self.goal_image,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        os.makedirs(self.inference.datastore_path_image, exist_ok=True)

    def _goal_pose_vec(self, x: float, y: float, yaw: float) -> np.ndarray:
        """World goal -> normalised [fwd, left, cos dh, sin dh] in the robot frame."""
        if self.goal["mode"] != "pose":
            return np.zeros(4, dtype=np.float32)
        dx, dy = self.goal["x"] - x, self.goal["y"] - y
        c, s = math.cos(yaw), math.sin(yaw)
        fwd, left = dx * c + dy * s, -dx * s + dy * c
        radius = math.hypot(fwd, left)
        if radius > self.thres_dist_m:
            fwd *= self.thres_dist_m / radius
            left *= self.thres_dist_m / radius
        dh = self.goal["heading_rad"] - yaw
        return np.array([fwd / self.waypoint_spacing, left / self.waypoint_spacing,
                         math.cos(dh), math.sin(dh)], dtype=np.float32)

    def step(self, img: Image.Image, x: float, y: float, yaw: float):
        goal_img = self.goal_image if self.goal_image is not None else img
        lan_inst = self.goal["instruction"] if ro.lan_prompt else "xxxx"
        batch = self.inference.data_transformer_omnivla(
            current_image_PIL=img,
            lan_inst=lan_inst,
            goal_image_PIL=goal_img,
            goal_pose_loc_norm=self._goal_pose_vec(x, y, yaw),
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        actions, _modality = self.inference.run_forward_pass(
            vla=self.vla.eval(), action_head=self.action_head.eval(),
            noisy_action_projector=None, pose_projector=self.pose_projector.eval(),
            batch=batch, action_tokenizer=self.action_tokenizer,
            device_id=self.device_id, use_l1_regression=True, use_diffusion=False,
            use_film=False, num_patches=self.num_patches, mode="vali",
        )
        chunk = actions.float().cpu().numpy()[0]          # (8, 4) normalised
        return chunk, self._twist(chunk)

    def _twist(self, chunk: np.ndarray) -> tuple[float, float]:
        """Chunk -> one (v, w) command, held until the next tick."""
        wp = chunk[self.waypoint_select].copy()
        wp[:2] *= self.waypoint_spacing
        dx, dy, hx, hy = wp
        eps = 1e-8
        if abs(dx) < eps and abs(dy) < eps:
            v, w = 0.0, math.atan2(hy, hx) / self.dt_s
        elif abs(dx) < eps:
            v, w = 0.0, math.copysign(math.pi / (2 * self.dt_s), dy)
        else:
            v, w = dx / self.dt_s, math.atan(dy / dx) / self.dt_s
        # Ratio-preserving clip: scale both so the commanded arc keeps its shape.
        if abs(v) <= self.max_v and abs(w) <= self.max_w:
            return float(v), float(w)
        scale = min(self.max_v / abs(v) if v else math.inf,
                    self.max_w / abs(w) if w else math.inf)
        return float(v * scale), float(w * scale)


class CCTVPolicy(OnboardOnlyPolicy):
    """OmniVLA with fixed external cameras as extra image slots (langsim recipe).

    Model layout: [ego_obs, ego_goal, cctv_0, cctv_1] through the SHARED ego
    encoder, each CCTV view avg-pooled 256 -> 64 tokens by a pixel-shuffle seam
    the trainer installs — so the seam must be installed here too, and
    num_patches recomputed (2*256 + 2*64 + 1 = 641), or the action hidden-state
    slice lands on the wrong tokens and the outputs are garbage without a crash.

    A missing CCTV frame in a request degrades gracefully: that slot is zeroed
    and its block_present flag cleared, exactly like training's dropped views.
    """

    def __init__(self, vla_path: str, resume_step: int, tick_hz: float,
                 max_v: float, max_w: float, waypoint_select: int,
                 spacing_m: float, cctv_cams: list[str],
                 downsample: tuple[str, int] = ("avgpool", 64),
                 thres_dist_m: float = 30.0):
        self.dt_s = 1.0 / tick_hz
        self.max_v, self.max_w = max_v, max_w
        self.waypoint_select = waypoint_select
        self.thres_dist_m = thres_dist_m
        self.waypoint_spacing = spacing_m
        self.cctv_cams = list(cctv_cams)          # slot order == training order

        n_img = 2 + len(self.cctv_cams)
        cfg = InferenceConfig(resume=True, vla_path=vla_path, resume_step=resume_step,
                              num_images_in_input=n_img)
        cfg.use_cctv_downsample = True            # -> define_model loads the v2 class
        (self.vla, self.action_head, self.pose_projector, self.device_id,
         self.num_patches, self.action_tokenizer, self.processor) = define_model(cfg)

        # install the avgpool seam exactly as train_omnivla_sim_cctv_v2.main does
        from prismatic.extern.hf.vision_encoder import build_cctv_downsampler
        method, out_tokens = downsample
        vb = self.vla.vision_backbone
        grid_side = int(round(vb.get_num_patches() ** 0.5))
        shuffle = build_cctv_downsampler(
            method, out_tokens, vision_dim=vb.embed_dim, grid_side=grid_side,
        ).to(dtype=torch.bfloat16, device=self.device_id)
        self.vla.install_pixel_shuffle_encoder(shuffle, 2, False,
                                               cctv_projector=None, cctv_cam_embed=None)
        # define_model computed the undownsampled count (n_img*256+1) — fix it.
        self.num_patches = 2 * vb.get_num_patches() + len(self.cctv_cams) * out_tokens + 1
        print(f"[server] CCTV mode: cams={self.cctv_cams}  {method}/{out_tokens}  "
              f"num_patches={self.num_patches}")
        self.inference = None
        self.goal = None
        self.goal_image = None

    def step(self, img: Image.Image, x: float, y: float, yaw: float,
             cctv: dict | None = None):
        cctv = cctv or {}
        goal_img = self.goal_image if self.goal_image is not None else img
        lan_inst = self.goal["instruction"] if ro.lan_prompt else "xxxx"
        # block_present must mirror the TRAINING rule (spot_real_cctv_dataset):
        #   ego_goal present only for image modalities (5, 6);
        #   pose block present only for pose modalities (4, 5, 8).
        # The cctv attention builder treats block_present as AUTHORITATIVE (no
        # modality AND), so language mode (7) must mask ego_goal AND pose here —
        # leaving them on attends a duplicate current image + a zero pose token
        # the model never saw at train time.
        ego_goal_present = 1.0 if ro.image_goal else 0.0
        pose_present = 1.0 if ro.pose_goal else 0.0
        ordered = [img, goal_img]
        present = [1.0, ego_goal_present]
        for cam in self.cctv_cams:
            raw = cctv.get(cam)
            if raw:
                ordered.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                present.append(1.0)
            else:
                ordered.append(Image.new("RGB", img.size))   # zeroed by transformer
                present.append(0.0)
        present.append(pose_present)                          # trailing pose block
        batch = self.inference.data_transformer_omnivla_cctv(
            ordered_images=ordered,
            block_present=np.asarray(present, dtype=np.float32),
            lan_inst=lan_inst,
            goal_pose_loc_norm=self._goal_pose_vec(x, y, yaw),
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        actions, _modality = self.inference.run_forward_pass(
            vla=self.vla.eval(), action_head=self.action_head.eval(),
            noisy_action_projector=None, pose_projector=self.pose_projector.eval(),
            batch=batch, action_tokenizer=self.action_tokenizer,
            device_id=self.device_id, use_l1_regression=True, use_diffusion=False,
            use_film=False, num_patches=self.num_patches, mode="vali",
        )
        chunk = actions.float().cpu().numpy()[0]
        return chunk, self._twist(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--vla-path", default=f"{_REPO_ROOT}/omnivla-original")
    ap.add_argument("--step", type=int, default=120000)
    ap.add_argument("--tick-hz", type=float, default=3.0)
    ap.add_argument("--max-v", type=float, default=0.3,
                    help="m/s SAFETY CAP on the commanded twist — not the denormaliser")
    ap.add_argument("--max-w", type=float, default=0.3, help="rad/s safety cap")
    ap.add_argument("--spacing", type=float, default=0.305,
                    help="metres per normalised waypoint unit (GNM mixture mean); "
                         "0.38 = SCAND/Spot, 0.333 = InfraVLA sim convention")
    ap.add_argument("--waypoint-select", type=int, default=4)
    ap.add_argument("--cctv", default="",
                    help="comma-separated CCTV cams in TRAINING slot order, e.g. "
                         "'camera3,camera4'. Empty = legacy ego-only policy.")
    args = ap.parse_args()

    # Bind BEFORE loading the model: the checkpoint takes ~60 s, and discovering
    # an "Address already in use" after that wastes the whole load. caladan is a
    # shared box, so port clashes with other users are routine.
    sock = zmq.Context().socket(zmq.REP)
    try:
        sock.bind(args.bind)
    except zmq.ZMQError as exc:
        raise SystemExit(
            f"[server] cannot bind {args.bind}: {exc}\n"
            f"          Another process holds that port (check: ss -lntp | grep "
            f"{args.bind.rsplit(':', 1)[-1]}).\n"
            f"          Pick a free one, e.g. --bind tcp://127.0.0.1:5601, and point\n"
            f"          the loop at it with -p server:=tcp://127.0.0.1:5601")
    print(f"[server] bound {args.bind} — loading the model (~60 s) ...")

    if args.cctv:
        cams = [c.strip() for c in args.cctv.split(",") if c.strip()]
        policy = CCTVPolicy(args.vla_path, args.step, args.tick_hz,
                            args.max_v, args.max_w, args.waypoint_select,
                            args.spacing, cctv_cams=cams)
    else:
        policy = OnboardOnlyPolicy(args.vla_path, args.step, args.tick_hz,
                                   args.max_v, args.max_w, args.waypoint_select,
                                   args.spacing)
    print(f"[server] denormalisation {args.spacing} m/unit; "
          f"safety caps v<={args.max_v} m/s, w<={args.max_w} rad/s")
    print("[server] model resident — ready for requests")
    while True:
        req = pickle.loads(sock.recv())
        try:
            if req["cmd"] == "info":
                # Identity of the model actually serving. Lets a run record name
                # its checkpoint instead of trusting a directory name typed by a
                # human, which is the only link we had before.
                sock.send(pickle.dumps({
                    "ok": True, "vla_path": args.vla_path, "step": args.step,
                    "spacing_m": policy.waypoint_spacing,
                    "cctv_cams": list(getattr(policy, "cctv_cams", [])),
                    "waypoint_select": policy.waypoint_select,
                    "tick_hz": round(1.0 / policy.dt_s, 4)}))
                continue
            if req["cmd"] == "reset":
                policy.reset(req["goal"])
                sock.send(pickle.dumps({"ok": True}))
                continue
            t0 = time.perf_counter()
            img = Image.open(io.BytesIO(req["image"])).convert("RGB")
            if isinstance(policy, CCTVPolicy):
                chunk, (v, w) = policy.step(img, req["x"], req["y"], req["yaw"],
                                            cctv=req.get("cctv"))
            else:
                chunk, (v, w) = policy.step(img, req["x"], req["y"], req["yaw"])
            sock.send(pickle.dumps({"ok": True, "v": v, "w": w,
                                    "waypoints": chunk.tolist(),      # NORMALISED units
                                    "spacing_m": policy.waypoint_spacing,
                                    "dt_s": time.perf_counter() - t0}))
        except Exception as exc:                       # never leave the REP socket hung
            import traceback; traceback.print_exc()
            sock.send(pickle.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
