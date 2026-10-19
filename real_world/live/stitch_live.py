
from __future__ import annotations

import math

import cv2
import numpy as np

CAMS = ("frontleft", "frontright")


def quat_matrix(q, t) -> np.ndarray:
    """(quaternion, translation) -> 4x4 homogeneous matrix."""
    x, y, z, w = q
    M = np.eye(4)
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    M[:3, 3] = t
    return M


class FrontStitcher:
    """Build once from rig geometry, then call composite() per frame."""

    def __init__(self, body_T_cam: dict, K: dict, src_hw: dict, *,
                 distance_m: float = 2.0, hfov_deg: float = 110.0,
                 width: int = 768, blend_power: float = 3.0, aspect: float = 4 / 3):
        axes = [body_T_cam[c][:3, :3] @ np.array([0, 0, 1.0]) for c in CAMS]
        # Virtual eye: MIDPOINT of the two cameras. BD's example sums the two
        # positions (`eye_wrt_body = pos1 + pos2`), which places the eye ~0.5 m
        # ahead of the real cameras and skews the effective FOV. Treated as a slip.
        eye = (body_T_cam[CAMS[0]][:3, 3] + body_T_cam[CAMS[1]][:3, 3]) / 2.0
        normal = axes[0] + axes[1]
        eye[1] = 0.0                      # centre it laterally
        normal[1] = 0.0
        normal /= np.linalg.norm(normal)
        left = np.cross([0, 0, 1.0], normal)
        left /= np.linalg.norm(left)
        up = np.cross(normal, left)
        up /= np.linalg.norm(up)

        H = int(round(width * 3 / 4))
        self.f = (width / 2) / math.tan(math.radians(hfov_deg / 2))
        sx, sy = np.meshgrid((np.arange(width) - (width - 1) / 2) / self.f,
                             (np.arange(H) - (H - 1) / 2) / self.f)
        # image +x is right (= -left), image +y is down (= -up)
        rays = (normal[None, None, :] + sx[..., None] * (-left)[None, None, :]
                + sy[..., None] * (-up)[None, None, :])
        pts = eye[None, None, :] + rays * (distance_m / (rays @ normal))[..., None]

        self.maps, self.weights = {}, {}
        for cam in CAMS:
            inv = np.linalg.inv(body_T_cam[cam])
            pc = pts @ inv[:3, :3].T + inv[:3, 3]
            z = pc[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = K[cam][0, 0] * pc[..., 0] / z + K[cam][0, 2]
                v = K[cam][1, 1] * pc[..., 1] / z + K[cam][1, 2]
            h, w = src_hw[cam]
            norm = np.stack([u / w, v / h], -1)
            ok = (z > 0) & (norm[..., 0] >= 0) & (norm[..., 0] < 1) \
                         & (norm[..., 1] >= 0) & (norm[..., 1] < 1)
            score = np.clip(1.0 - np.abs(norm * 2.0 - 1.0), 0, None)
            self.weights[cam] = np.where(
                ok, np.power(np.clip(score[..., 0] * score[..., 1] * 0.999 + 0.001, 0, 1),
                             blend_power), 0.0)
            self.maps[cam] = (np.nan_to_num(u).astype(np.float32),
                              np.nan_to_num(v).astype(np.float32))

        self.overlap = (self.weights[CAMS[0]] > 1e-6) & (self.weights[CAMS[1]] > 1e-6)
        total = self.weights[CAMS[0]] + self.weights[CAMS[1]]
        self.crop = self._inscribed(total > 0, aspect)
        x0, y0, x1, y1 = self.crop
        self.out_size = (x1 - x0, y1 - y0)
        self.hfov = 2 * math.degrees(math.atan((x1 - x0) / 2 / self.f))
        self.vfov = 2 * math.degrees(math.atan((y1 - y0) / 2 / self.f))

    @staticmethod
    def _inscribed(valid: np.ndarray, aspect: float):
        H, W = valid.shape
        ys, xs = np.where(valid)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        ii = np.pad(np.cumsum(np.cumsum(valid.astype(np.int32), 0), 1), ((1, 0), (1, 0)))

        def full(x0, y0, x1, y1):
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 <= x0 or y1 <= y0:
                return False
            return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0] == (x1 - x0) * (y1 - y0)

        lo, hi = 0.0, float(min(W / aspect, H))
        for _ in range(45):
            h = (lo + hi) / 2
            if full(round(cx - h * aspect / 2), round(cy - h / 2),
                    round(cx + h * aspect / 2), round(cy + h / 2)):
                lo = h
            else:
                hi = h
        w = lo * aspect
        return (round(cx - w / 2), round(cy - lo / 2), round(cx + w / 2), round(cy + lo / 2))

    def composite(self, frames: dict) -> np.ndarray:
        """{'frontleft': HxWx3 RGB, 'frontright': ...} -> stitched RGB, cropped."""
        warped = {c: cv2.remap(frames[c], *self.maps[c], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
                  for c in CAMS}
        if self.overlap.any():
            # Spot's front cameras auto-expose independently — on this rig the
            # right ran ~27% brighter. Without this the seam is a visible step.
            ref = warped[CAMS[0]][self.overlap].mean(0)
            gain = ref / np.maximum(warped[CAMS[1]][self.overlap].mean(0), 1e-6)
            warped[CAMS[1]] = np.clip(warped[CAMS[1]] * np.clip(gain, 0.2, 5.0), 0, 255)
        total = self.weights[CAMS[0]] + self.weights[CAMS[1]]
        safe = np.where(total == 0, 1, total)
        out = sum(warped[c] * (self.weights[c] / safe)[..., None] for c in CAMS)
        x0, y0, x1, y1 = self.crop
        return np.clip(out, 0, 255).astype(np.uint8)[y0:y1, x0:x1]

    def project(self, bearing_deg: float) -> float:
        """Bearing (deg, + = left) -> image column in the cropped output."""
        return self.out_size[0] / 2 - self.f * math.tan(math.radians(bearing_deg))
