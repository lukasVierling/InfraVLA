
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# diag(1, -1, -1): expresses an OpenCV-cam axis triple in USD-cam axes (flip Y,Z).
_USD_FROM_CV = np.diag([1.0, -1.0, -1.0])


def intrinsics_matrix(
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    vertical_aperture_mm: float,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    """USD aperture-form intrinsics -> 3x3 pin-hole matrix K.

    USD stores a physical focal length and sensor aperture (both in mm); the
    pixel focal length is `f_px = f_mm / aperture_mm * size_px`. USD also assumes
    a centred principal point, so (cx, cy) = (W/2, H/2).

    e.g. cctv_topdown.yaml: 18.148 mm / 20.955 mm * 1280 px -> fx ~= 1108 px.

    Returns
        K : (3, 3) float64.  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    fx = focal_length_mm / horizontal_aperture_mm * width_px
    fy = focal_length_mm / vertical_aperture_mm * height_px
    cx = width_px / 2.0
    cy = height_px / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_world_from_cv(euler_xyz_deg) -> np.ndarray:
    """Isaac `rotateXYZ` Euler triple (deg) -> R mapping OpenCV-cam -> world.

    Two steps:
      1. USD rotateXYZ == SciPy extrinsic "xyz"  -> R_world_from_usd.
      2. flip to OpenCV axes                      -> @ diag(1, -1, -1).

    The columns of the returned matrix are the OpenCV camera's (x_right, y_down,
    z_forward) axes expressed in world coordinates.

    Args
        euler_xyz_deg : (3,) [rx, ry, rz] degrees, as in the yaml.
    Returns
        R_wc : (3, 3) float64.  world_point = R_wc @ cam_point + t
    """
    r_world_from_usd = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_matrix()
    return r_world_from_usd @ _USD_FROM_CV


@dataclass
class CameraModel:
    """A static pin-hole camera: intrinsics K, pose (R_wc, t) in the world frame.

    R_wc maps OpenCV-cam coords -> world coords; t is the camera centre (m).
    All projection helpers assume the OpenCV convention (+Z in front of the cam).
    """

    K: np.ndarray            # (3, 3)
    R_wc: np.ndarray         # (3, 3) world_from_cv
    t: np.ndarray            # (3,)   camera centre in world

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_usd(
        cls,
        focal_length_mm: float,
        horizontal_aperture_mm: float,
        vertical_aperture_mm: float,
        width_px: int,
        height_px: int,
        world_translate_m,
        world_rotate_euler_deg,
    ) -> "CameraModel":
        """Build straight from the fields of an Isaac camera yaml + image size."""
        K = intrinsics_matrix(focal_length_mm, horizontal_aperture_mm,
                              vertical_aperture_mm, width_px, height_px)
        R_wc = rotation_world_from_cv(world_rotate_euler_deg)
        t = np.asarray(world_translate_m, dtype=np.float64)
        return cls(K=K, R_wc=R_wc, t=t)

    # --------------------------------------------------------------- queries
    @property
    def R_cw(self) -> np.ndarray:
        """world -> OpenCV-cam rotation (the transpose of R_wc)."""
        return self.R_wc.T

    def project_points(self, points_world: np.ndarray):
        """World points -> pixel (u, v) + camera-frame depth z.

        Args
            points_world : (..., 3) world coordinates (m).
        Returns
            uv    : (..., 2) pixel coords (float; not rounded, not clipped).
            z_cam : (...,)   depth along the cam +Z axis. z_cam > 0 == in front.

        Worked shape: points_world (H, W, 3) -> uv (H, W, 2), z_cam (H, W).
        """
        p = np.asarray(points_world, dtype=np.float64)
        # cam coords: p_cam = R_cw @ (P - t).  Vectorised over leading dims:
        p_cam = (p - self.t) @ self.R_cw.T          # (..., 3)
        z_cam = p_cam[..., 2]
        # Pin-hole projection. Guard z==0 so the divide can't blow up; those
        # points are flagged invalid by the z_cam>0 test the caller applies.
        z_safe = np.where(np.abs(z_cam) < 1e-9, 1e-9, z_cam)
        x_n = p_cam[..., 0] / z_safe
        y_n = p_cam[..., 1] / z_safe
        u = self.K[0, 0] * x_n + self.K[0, 2]
        v = self.K[1, 1] * y_n + self.K[1, 2]
        uv = np.stack([u, v], axis=-1)              # (..., 2)
        return uv, z_cam

    def back_project_to_ground(self, uv: np.ndarray, max_range_m: float | None = None):
        """Pixel (u, v) -> the world point where its ray hits the floor z = 0.

        Casts the pin-hole ray for each pixel and intersects it with the ground
        plane. A pixel looking at/above the horizon never hits the floor in front
        of the camera; those are returned with valid=False.

        Args
            uv          : (..., 2) pixel coords.
            max_range_m : if set, hits farther than this (horizontal distance
                          from the camera) are marked invalid — keeps the
                          near-horizon tail (which projects to ~infinity) from
                          dominating an auto-computed map extent.
        Returns
            xy    : (..., 2) world (X, Y) of the floor hit (NaN where invalid).
            valid : (...,) bool.
        """
        uv = np.asarray(uv, dtype=np.float64)
        ones = np.ones(uv.shape[:-1] + (1,))
        pix_h = np.concatenate([uv, ones], axis=-1)          # (..., 3) [u, v, 1]
        # ray direction in cam frame, then rotate into world.
        d_cam = pix_h @ np.linalg.inv(self.K).T              # (..., 3)
        d_world = d_cam @ self.R_wc.T                        # (..., 3)
        dz = d_world[..., 2]
        # floor hit: t.z + s*dz = 0  ->  s = -t.z / dz, must have s > 0.
        dz_safe = np.where(np.abs(dz) < 1e-9, 1e-9, dz)
        s = -self.t[2] / dz_safe
        valid = (s > 0) & (np.abs(dz) >= 1e-9)
        hit = self.t + s[..., None] * d_world                # (..., 3)
        xy = hit[..., :2]
        if max_range_m is not None:
            horiz = np.linalg.norm(hit[..., :2] - self.t[:2], axis=-1)
            valid = valid & (horiz <= max_range_m)
        xy = np.where(valid[..., None], xy, np.nan)
        return xy, valid

    def ground_homography(self) -> np.ndarray:
        """3x3 homography H mapping floor [X, Y, 1] -> pixel [u, v, 1] (up to scale).

        For a ground point Z=0, p_cam = R_cw[:,0]*X + R_cw[:,1]*Y - R_cw@t, so the
        third column of R_cw (which multiplies Z) drops out and projection becomes
        a plain 3x3. Handy for tests and for warping whole images; the BEV mapper
        itself uses project_points (an inverse warp) instead.
        """
        Rcw = self.R_cw
        cols = np.column_stack([Rcw[:, 0], Rcw[:, 1], -Rcw @ self.t])  # (3, 3)
        return self.K @ cols
