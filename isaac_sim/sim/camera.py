import os
import numpy as np

# NOTE: the isaacsim Camera import requires SimulationApp to already be booted in this process.
from isaacsim.sensors.camera import Camera as IsaacCamera
from pxr import Gf, UsdGeom


class Camera:
    """Create a single camera isntance in an isaac sim env."""

    def destroy(self):
        """Destroy the camera in the isaac sim env."""
        if getattr(self, "cam", None) is not None:
            self.cam.destroy()
            self.cam = None

    def __init__(self, cfg, stage, world=None):
        """
        Create a camer instance in isaac sim.
        
        Args:
            cfg: The camera config file that contains most important parameters
            stage: The usd stage of the issac sim env, camera will be attached here
            world: The world isntance of the env which can be used to step until the cameras show non black output
        """
        self.cfg = cfg
        self.world = world

        # Auto-discover the camera prim when no exact prim_path is given.
        prim_path = cfg.prim_path
        if prim_path is None:
            ad = cfg.auto_discover
            matches = [
                str(p.GetPath()) for p in stage.Traverse()
                if str(p.GetPath()).startswith(ad.parent_prefix) and p.IsA(UsdGeom.Camera)
            ]
            if not matches:
                raise RuntimeError(
                    f"[Camera {cfg.name!r}] no camera prims found under "
                    f"{ad.parent_prefix!r}. Check the robot USD actually loaded "
                    f"and that env.robot.prim_path is correct."
                )
            # Print all candidates so the user can pick a disambiguating hint.
            print(
                f"[Camera {cfg.name!r}] auto-discover under {ad.parent_prefix!r} "
                f"found {len(matches)} camera prims:"
            )
            for m in matches:
                print(f"    {m}")
            hits = [p for p in matches if any(k in p.lower() for k in ad.name_hints)]
            if not hits:
                # Refuse to guess: silently taking matches[0] can bind the wrong camera.
                raise RuntimeError(
                    f"[Camera {cfg.name!r}] none of the name_hints {list(ad.name_hints)} "
                    f"matched any camera prim under {ad.parent_prefix!r}. "
                    f"Available prims printed above — pick a substring that "
                    f"appears in exactly one of them, OR set `prim_path` "
                    f"explicitly in the camera yaml to bypass auto-discovery."
                )
            if len(hits) > 1:
                print(
                    f"[Camera {cfg.name!r}] WARNING: name_hints matched "
                    f"{len(hits)} prims; using the first ({hits[0]}). "
                    f"Tighten the hint or set prim_path explicitly for a "
                    f"deterministic pick."
                )
            prim_path = hits[0]
            print(f"[Camera {cfg.name!r}] using prim_path = {prim_path}")

        if cfg.translation_required:
            extr = cfg.extrinsics
            if "world_transform_matrix" in extr:
                # CCTV-style with a VERBATIM 4x4 pose matrix (row-major, as saved
                # from a GUI stage export) — no euler conversion, no convention risk.
                prim = UsdGeom.Camera.Define(stage, prim_path)
                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                m = [float(v) for row in extr.world_transform_matrix for v in row]
                xform.AddTransformOp().Set(Gf.Matrix4d(*m))
                api = UsdGeom.Camera(prim)
                if cfg.intrinsics.focal_length_mm is not None:
                    api.GetFocalLengthAttr().Set(float(cfg.intrinsics.focal_length_mm))
                if cfg.intrinsics.horizontal_aperture_mm is not None:
                    api.GetHorizontalApertureAttr().Set(float(cfg.intrinsics.horizontal_aperture_mm))
                if cfg.intrinsics.vertical_aperture_mm is not None:
                    api.GetVerticalApertureAttr().Set(float(cfg.intrinsics.vertical_aperture_mm))
            elif "world_translate_m" in extr:
                # CCTV-style: define a brand-new camera prim at a world pose.
                prim = UsdGeom.Camera.Define(stage, prim_path)
                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*extr.world_translate_m))
                xform.AddRotateXYZOp().Set(Gf.Vec3f(*extr.world_rotate_euler_deg))
                api = UsdGeom.Camera(prim)
                if cfg.intrinsics.focal_length_mm is not None:
                    api.GetFocalLengthAttr().Set(float(cfg.intrinsics.focal_length_mm))
                if cfg.intrinsics.horizontal_aperture_mm is not None:
                    api.GetHorizontalApertureAttr().Set(float(cfg.intrinsics.horizontal_aperture_mm))
                if cfg.intrinsics.vertical_aperture_mm is not None:
                    api.GetVerticalApertureAttr().Set(float(cfg.intrinsics.vertical_aperture_mm))
            else:
                # Chase-style: reuse the existing prim, add a local translate op.
                prim = stage.GetPrimAtPath(prim_path)
                xform = UsdGeom.Xformable(prim)
                translate_op = next(
                    (op for op in xform.GetOrderedXformOps()
                     if op.GetOpName() == "xformOp:translate"), None,
                )
                if translate_op is None:
                    translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(*extr.local_translate_m))

        self.cam = IsaacCamera(prim_path=prim_path, name=cfg.name, resolution=tuple(cfg.resolution), frequency=cfg.frequency)
        self.cam.initialize()
        if int(os.environ.get("ISAAC_RENDER_FLUSH_K", "0")) > 0:
            # Flush-render mode (see isaac_sim_env.step banner): physics-free
            # World.render() flushes carry a NON-advancing sim-time stamp, so the
            # camera's `elapsed >= 1/frequency` acquisition gate never passes and
            # get_rgba() stays stale forever. -1 = acquire EVERY rendered frame
            # (bypasses the sim-time gate; harmless in this mode since renders
            # only happen right before a readback anyway).
            self.cam.set_frequency(-1)

    def capture(self):
        """Captures a frame of the camera

        Returns:
            A (H, W, 3) uint8 RGB image from the camera. May
        """
        frame = self.cam.get_rgba()
        if (frame is None or frame.size == 0) and self.world is not None:
            self.world.step(render=True)
            frame = self.cam.get_rgba()
        if frame is None or frame.size == 0:
            raise RuntimeError("No frame captured by camera.")
        return frame[..., :3].astype(np.uint8)
