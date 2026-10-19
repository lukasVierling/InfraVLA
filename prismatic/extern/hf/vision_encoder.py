"""Swappable vision-encoder seam for the OmniVLA fork.

WHY THIS EXISTS
---------------
Today the model's `forward()` calls the vision backbone + projector inline and
the CCTV attention-mask builder *recomputes* the per-view token count as a
hardcoded ``[P] * n_img`` (P == 256). That hardcoding is the single thing that
blocks swapping the encoder: a pixel-shuffled CCTV stream emits 64 tokens/view,
VGGT emits a variable patch grid, a map encoder emits something else entirely.

This module introduces one explicit boundary so the encoder becomes a config
choice instead of inline code:

  * ``EncodedVision``      — the struct the LLM-assembly side gets back. The only
                            load-bearing field today is ``tokens`` (identical to
                            what ``_process_vision_features`` returned before).
                            ``per_block_widths`` *replaces* the hardcoded
                            ``[P]*n_img`` so a swapped-in encoder may emit a
                            different token count per view with NO edit to the
                            mask builder.
  * ``VisionEncoder``      — a transport-neutral Protocol. In-process and (future)
                            remote impls share this signature, so the VLA forward
                            never branches on transport.
  * ``LocalDinoSigLipEncoder`` — the default in-process impl. A thin facade over
                            the EXISTING backbone + projector. It is NOT an
                            ``nn.Module`` and owns no parameters, so it adds no
                            state_dict keys and no LoRA targets — the seam is
                            byte-identical to the pre-seam code path.

Design decisions locked with Lukas (2026-06-21):
  * In-process default returns an *attached* tensor → autograd intact →
    LoRA-on-cctv_featurizer keeps training. Remote/frozen is a separate opt-in
    impl behind this same Protocol, inference-only.
  * ``camera_params`` enters the contract now (optional) but the default encoder
    IGNORES it — turning on per-camera extrinsics injection later needs no
    signature change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, Literal, runtime_checkable

import torch
import torch.nn as nn

# A per-view tag distinguishing the robot's own cameras from fixed CCTV views.
SlotKind = Literal["ego", "cctv"]


@dataclass(frozen=True)
class EncodedVision:
    """Everything the LLM-assembly side needs back from a vision encoder.

    Fields
    ------
    tokens : (B, T_total, llm_dim)
        Projected, LLM-width vision tokens. T_total == per_block_widths.sum().
        For the current encoder this is the EXACT tensor
        ``_process_vision_features`` produced before the seam: (B, n_img*256, 4096).
        Attached to the autograd graph for the in-process default; a remote impl
        would return a detached tensor (and thus a frozen encoder).
    per_block_widths : (num_blocks,) long
        Token count per IMAGE block, in emission order. ``[256, 256, 256]`` today;
        e.g. ``[256, 256, 64, 64]`` once CCTV views are pixel-shuffled. This is
        what replaces the hardcoded ``[P]*n_img`` in the CCTV mask builder. It
        does NOT include the trailing pose/proprio token (the caller appends that).
    block_present : (B, num_blocks + 1) or None
        Pass-through of the per-block presence mask (image blocks + the trailing
        pose flag) so the downstream CCTV attention builder consumes it unchanged.
        None on the legacy non-CCTV path.
    metadata : open dict
        Channel for encoder-specific extras a future encoder may surface without
        changing this dataclass: e.g. VGGT's predicted camera poses / depth maps.
        Empty ``{}`` for the current encoder.
    """

    tokens: torch.Tensor
    per_block_widths: torch.Tensor
    block_present: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Cheap invariant: the token count must equal the widths the mask builder
        # will expand over. Catches a desynced encoder BEFORE it silently
        # corrupts the attention mask / label fill. Mirrors the existing assert
        # at modeling_prismatic.py (the cctv builder).
        n_tokens = self.tokens.shape[1]
        n_widths = int(self.per_block_widths.sum())
        assert n_tokens == n_widths, (
            f"EncodedVision: tokens length {n_tokens} != sum(per_block_widths) {n_widths}"
        )
        if self.block_present is not None:
            # block_present carries one flag per image block PLUS the trailing
            # pose/proprio flag, hence num_blocks + 1.
            n_blocks = self.per_block_widths.numel()
            assert self.block_present.shape[1] == n_blocks + 1, (
                f"EncodedVision: block_present width {self.block_present.shape[1]} "
                f"!= num_blocks + 1 ({n_blocks + 1})"
            )

    def expand_image_presence_to_tokens(self, dtype: torch.dtype) -> torch.Tensor:
        """Expand ``block_present`` (B, num_blocks+1) -> per-token mask (B, T_total+1).

        Image-block flags expand over ``per_block_widths``; the final pose/proprio
        flag (width 1) is appended as-is. This reproduces the inline
        ``widths=[P]*n_img+[1]; repeat_interleave(...)`` the CCTV mask builder does,
        but driven by the encoder-reported widths instead of a constant. Provided
        as a reusable helper (and a unit-test target); the mask builder may call
        it or keep its own repeat_interleave — either way the numbers match.
        """
        assert self.block_present is not None, "need block_present to expand"
        image_widths = self.per_block_widths  # (num_blocks,)
        pose_width = torch.ones(1, dtype=image_widths.dtype, device=image_widths.device)
        all_widths = torch.cat([image_widths, pose_width])  # (num_blocks + 1,)
        return torch.repeat_interleave(self.block_present.to(dtype), all_widths, dim=1)


@runtime_checkable
class VisionEncoder(Protocol):
    """Transport-neutral vision-encoder contract.

    In-process and (future) remote impls share THIS signature, so the VLA forward
    and the eval client never branch on transport — swapping encoders is a config
    flag, not a call-site change. The static ``vision_dim`` / ``llm_dim``
    properties let the projector be sized and a remote client allocate buffers
    WITHOUT instantiating weights.
    """

    @property
    def vision_dim(self) -> int:
        """Pre-projection feature width (2176 fused = 1024 DINOv2 + 1152 SigLIP; 2048 VGGT)."""
        ...

    @property
    def llm_dim(self) -> int:
        """Post-projection token width handed to the LLM (4096)."""
        ...

    def tokens_per_image(self, slot_kind: SlotKind) -> int:
        """Tokens this encoder emits for one view of the given kind (256 today)."""
        ...

    def encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,        # (B, n_img*6, 224, 224) bf16 — in-process path
        images: Optional[Any] = None,                       # (B, N, H, W, 3) uint8 — remote/VGGT path
        slot_kinds: Optional[Sequence[SlotKind]] = None,    # None => derive from num_ego_images
        block_present: Optional[torch.Tensor] = None,       # (B, n_img+1) pass-through to the mask builder
        camera_params: Optional[dict[str, torch.Tensor]] = None,  # MAY be ignored by an impl
        language_embeddings: Optional[torch.Tensor] = None, # only used when use_film=True
        use_film: bool = False,
    ) -> EncodedVision:
        ...


class LocalDinoSigLipEncoder:
    """Default in-process encoder: a thin facade over the EXISTING
    ``PrismaticVisionBackbone`` + ``PrismaticProjector``.

    Zero behavioural change: ``.encode(pixel_values).tokens`` is byte-identical to
    the pre-seam ``_process_vision_features`` output.

    It holds REFERENCES (not copies) to the live modules and is deliberately NOT
    an ``nn.Module`` — so the model's ``state_dict`` keys, ``wrap_lora`` targets,
    and ``add_cctv_encoder`` ordering are all completely unaffected, and the
    returned tensor stays attached to the autograd graph (LoRA keeps training).
    Ego/CCTV routing already lives inside the backbone's index loop, so
    ``slot_kinds`` is accepted for forward-compat but unused here.
    """

    def __init__(self, vision_backbone, projector):
        self._backbone = vision_backbone
        self._projector = projector

    @property
    def vision_dim(self) -> int:
        return self._backbone.embed_dim  # 2176 fused

    @property
    def llm_dim(self) -> int:
        return self._projector.llm_dim  # 4096

    def tokens_per_image(self, slot_kind: SlotKind) -> int:
        # Uniform (256) for ego and cctv today. The arg exists so a pixel-shuffle
        # impl can return e.g. 64 for 'cctv' without an API change.
        return self._backbone.get_num_patches()

    def encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        images: Optional[Any] = None,                       # unused by the in-process impl
        slot_kinds: Optional[Sequence[SlotKind]] = None,    # unused: backbone routes by index
        block_present: Optional[torch.Tensor] = None,
        camera_params: Optional[dict[str, torch.Tensor]] = None,  # intentionally ignored for now
        language_embeddings: Optional[torch.Tensor] = None,
        use_film: bool = False,
    ) -> EncodedVision:
        assert pixel_values is not None, "LocalDinoSigLipEncoder needs pixel_values"

        # --- the exact two lines the pre-seam _process_vision_features ran ---
        if use_film:
            # FiLM: infuse language into visual features (OpenVLA-OFT path).
            patch_features = self._backbone(pixel_values, language_embeddings)  # (B, n_img*256, 2176)
        else:
            patch_features = self._backbone(pixel_values)                       # (B, n_img*256, 2176)
        tokens = self._projector(patch_features)                               # (B, n_img*256, 4096)

        # --- per-block widths: [256]*n_img today (uniform); the field is what
        #     lets a future encoder report a non-uniform layout instead. ---
        n_img = self._backbone.get_num_images_in_input()
        P = self._backbone.get_num_patches()  # 256
        per_block_widths = torch.full((n_img,), P, dtype=torch.long, device=tokens.device)

        return EncodedVision(
            tokens=tokens,
            per_block_widths=per_block_widths,
            block_present=block_present,  # forwarded untouched to the CCTV mask builder
            metadata={},
        )


class CCTVAvgPool(nn.Module):
    """PARAMETER-FREE 2D average-pool token downsample for ONE CCTV view. Reshapes
    the (grid x grid) patch grid and average-pools by an integer factor f (f=2 -> 64
    tokens, f=4 -> 16 tokens): each output token is the mean of an f x f patch
    neighbourhood. The learnable counterpart is CCTVConv (same spatial grouping, a
    depthwise conv init to this mean). Channel dim preserved (C in -> C out) so the
    projector consumes it unchanged. Has NO learnable params
    (next(self.parameters()) is empty) — nothing to train/save/load.

    Interface: out_tokens attr + forward (B, grid^2, C) -> (B, out_tokens, C).
    """

    def __init__(self, vision_dim: int, grid_side: int = 16, out_tokens: int = 64):
        super().__init__()
        out_side = int(round(out_tokens ** 0.5))
        assert out_side * out_side == out_tokens, f"out_tokens {out_tokens} must be a perfect square"
        assert grid_side % out_side == 0, f"grid {grid_side} not divisible by out_side {out_side}"
        self.vision_dim = vision_dim
        self.grid_side = grid_side
        self.out_tokens = out_tokens
        self.factor = grid_side // out_side                  # 2 (->64) or 4 (->16) for a 16x16 grid
        self.pool = nn.AvgPool2d(kernel_size=self.factor, stride=self.factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # x: (B, grid^2, C)
        B, _, C = x.shape
        h = self.grid_side
        x = x.view(B, h, h, C).permute(0, 3, 1, 2)            # (B, C, h, h)
        x = self.pool(x)                                      # (B, C, h/f, h/f)
        x = x.permute(0, 2, 3, 1).reshape(B, self.out_tokens, C)  # (B, (h/f)^2, C)
        return x


class CCTVConv(nn.Module):
    """LEARNABLE depthwise 2D conv token downsample for ONE CCTV view — the minimal
    learnable generalization of CCTVAvgPool. Identical reshape + spatial grouping;
    the only change is AvgPool2d (fixed 1/f^2 weights) -> a depthwise Conv2d
    (groups=C, so a per-channel f x f kernel, NO cross-channel mixing). Params =
    C * f * f (+ C bias); e.g. 2176 * 2*2 = 8704 for f=2 (->64). The conv weight is
    INITIALISED to the average-pool (uniform 1/f^2, zero bias) so at step 0 it is
    bit-identical to CCTVAvgPool and learns away from it — keeps the avgpool64 vs
    conv64 comparison clean. Channel dim preserved (C -> C).

    Interface mirrors CCTVAvgPool: out_tokens attr + forward (B, grid^2, C) ->
    (B, out_tokens, C).
    """

    def __init__(self, vision_dim: int, grid_side: int = 16, out_tokens: int = 64):
        super().__init__()
        out_side = int(round(out_tokens ** 0.5))
        assert out_side * out_side == out_tokens, f"out_tokens {out_tokens} must be a perfect square"
        assert grid_side % out_side == 0, f"grid {grid_side} not divisible by out_side {out_side}"
        self.vision_dim = vision_dim
        self.grid_side = grid_side
        self.out_tokens = out_tokens
        self.factor = grid_side // out_side                  # 2 (->64) or 4 (->16) for a 16x16 grid
        # depthwise: groups == channels -> one f x f spatial kernel per channel
        self.conv = nn.Conv2d(vision_dim, vision_dim, kernel_size=self.factor,
                              stride=self.factor, groups=vision_dim, bias=True)
        with torch.no_grad():                                # init == average pool (1/f^2, zero bias)
            self.conv.weight.fill_(1.0 / (self.factor * self.factor))
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, grid^2, C)
        B, _, C = x.shape
        h = self.grid_side
        x = x.view(B, h, h, C).permute(0, 3, 1, 2)           # (B, C, h, h)
        x = self.conv(x)                                     # (B, C, h/f, h/f)
        x = x.permute(0, 2, 3, 1).reshape(B, self.out_tokens, C)  # (B, (h/f)^2, C)
        return x


def build_cctv_downsampler(method: str, out_tokens: int, vision_dim: int, grid_side: int = 16):
    """Factory for the CCTV token-downsampler (single source of truth for train + eval).

    method: 'none' -> None (no downsampling); 'avgpool' -> CCTVAvgPool (param-free);
    'conv' -> CCTVConv (learnable depthwise pooling, ~8.7K params, init=avgpool).
    out_tokens: 64 or 16. Returned modules share the interface
    (out_tokens attr + forward (B, grid^2, C) -> (B, out_tokens, C)).
    """
    method = (method or "none").lower()
    if method in ("none", ""):
        return None
    if method == "avgpool":
        return CCTVAvgPool(vision_dim=vision_dim, grid_side=grid_side, out_tokens=out_tokens)
    if method == "conv":
        return CCTVConv(vision_dim=vision_dim, grid_side=grid_side, out_tokens=out_tokens)
    raise ValueError(f"unknown cctv downsample method '{method}' (expected none|avgpool|conv)")


def resolve_cctv_downsample(get):
    """Resolve (method, out_tokens) from a config accessor `get(key, default)`,
    honouring the `cctv_downsample: {method, out_tokens}` block. Returns ('none', 0)
    when absent.
    """
    ds = get("cctv_downsample", None)
    if ds:
        return str(ds["method"]), int(ds["out_tokens"])
    return "none", 0


def resolve_image_slot_perm(image_slot_order, num_ego_images, cctv_cam_names):
    """Resolve cfg `image_slot_order` (list of slot NAMES) -> integer permutation
    over the encoder's per-image blocks. SINGLE SOURCE OF TRUTH for train + eval,
    so the deployed token order is byte-identical to training (a mismatch would
    silently scramble the LLM's view of the world).

    Canonical (physical / encode) order is ego-first — the order the collator stacks
    images and the backbone routes them:
        ["ego_obs","ego_goal"][:num_ego_images] + list(cctv_cam_names)

    Returns perm where perm[k] = canonical block index that lands at output pos k,
    or None when image_slot_order is None (=> identity => legacy behaviour).
    Raises (NO silent fallback) if image_slot_order isn't an exact permutation of
    the canonical names — a typo'd / missing / extra cam must fail loudly.
    """
    if image_slot_order is None:
        return None
    ego_names = ["ego_obs", "ego_goal"][:num_ego_images]
    canonical = ego_names + list(cctv_cam_names)
    order = list(image_slot_order)
    if sorted(order) != sorted(canonical):
        raise ValueError(
            f"image_slot_order {order} must be a permutation of the canonical "
            f"slot names {canonical} (ego-first). Check num_ego_images / cctv_cam_names."
        )
    return [canonical.index(name) for name in order]


def permute_token_blocks(tokens, per_block_widths, perm):
    """Reorder per-image token blocks + their widths by `perm` (perm[k] = src block
    index at output position k). Identity when perm is None. WIDTH-AWARE (uses the
    real per-block widths), so it is correct for the downsampling encoders too, not
    just the uniform full-token case.
        tokens: (B, T_total, C) with T_total == per_block_widths.sum()
    """
    if perm is None:
        return tokens, per_block_widths
    blocks = list(torch.split(tokens, per_block_widths.tolist(), dim=1))  # n_img x (B, w_i, C)
    tokens = torch.cat([blocks[i] for i in perm], dim=1)                  # reorder + re-concat
    per_block_widths = per_block_widths[list(perm)]
    return tokens, per_block_widths


def permute_block_present(block_present, n_img, perm):
    """Reorder the `n_img` image columns of block_present by `perm`; the trailing
    pose/proprio flag (column n_img) stays LAST. Identity when perm/bp is None.
        block_present: (B, n_img + 1)
    """
    if perm is None or block_present is None:
        return block_present
    img = block_present[:, :n_img][:, list(perm)]            # (B, n_img) reordered
    return torch.cat([img, block_present[:, n_img:]], dim=1)  # keep pose flag last


class PixelShuffleDinoSigLipEncoder:
    """Like LocalDinoSigLipEncoder, but pixel-shuffles the CCTV views' tokens
    (256 -> 16) before projection, leaving the EGO views at full 256. Reports a
    non-uniform per_block_widths = [256]*n_ego + [out_tokens]*n_cctv, which the CCTV
    mask builder consumes directly (no mask-builder change needed — that is the whole
    point of returning per_block_widths from the encoder).

    The shuffle module is a STANDALONE nn.Module passed in (NOT a vla submodule), so
    wrap_lora — which only scans the model's own submodules — never sees it. It is
    therefore trained FULLY (random-init linears need full training, not a low-rank
    adapter) and saved/loaded as a peer of pose_projector / action_head. This encoder
    holds a plain reference to it, which stays valid because nobody re-wraps a
    standalone module (unlike LoRA-targeted modules whose leaves get replaced).
    """

    def __init__(self, vision_backbone, projector, cctv_pixel_shuffle, num_ego_images: int,
                 downsample_ego: bool = False,
                 cctv_projector=None, cctv_cam_embed=None):
        self._backbone = vision_backbone
        self._projector = projector
        self._shuffle = cctv_pixel_shuffle
        self._num_ego = num_ego_images
        self._downsample_ego = downsample_ego   # True => pool ego views too (uniform downsample)
        # Optional CCTV-specific extras (both default None => byte-identical to the
        # shared-projector path):
        #   cctv_projector : distinct vision->LLM MLP for CCTV blocks (warm-started
        #                    copy of the ego projector; own LoRA). None => shared.
        #   cctv_cam_embed : (num_cctv, llm_dim) learned per-camera identity vector
        #                    added to each CCTV block (zero-init => no-op at start).
        self._cctv_projector = cctv_projector
        self._cctv_cam_embed = cctv_cam_embed

    @property
    def vision_dim(self) -> int:
        return self._backbone.embed_dim

    @property
    def llm_dim(self) -> int:
        return self._projector.llm_dim

    def tokens_per_image(self, slot_kind: SlotKind) -> int:
        if slot_kind == "cctv" or self._downsample_ego:
            return self._shuffle.out_tokens                   # downsampled view -> out_tokens
        return self._backbone.get_num_patches()               # 256 (full-res ego)

    def encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        images: Optional[Any] = None,
        slot_kinds: Optional[Sequence[SlotKind]] = None,
        block_present: Optional[torch.Tensor] = None,
        camera_params: Optional[dict[str, torch.Tensor]] = None,  # ignored for now
        language_embeddings: Optional[torch.Tensor] = None,
        use_film: bool = False,
    ) -> EncodedVision:
        assert pixel_values is not None, "PixelShuffleDinoSigLipEncoder needs pixel_values"

        if use_film:
            patch_features = self._backbone(pixel_values, language_embeddings)
        else:
            patch_features = self._backbone(pixel_values)     # (B, n_img*256, C)

        n_img = self._backbone.get_num_images_in_input()
        P = self._backbone.get_num_patches()                  # 256
        blocks = patch_features.split(P, dim=1)               # n_img tuples of (B, 256, C)

        # CCTV blocks (slots >= num_ego) get downsampled; ego blocks too when
        # downsample_ego (uniform: every view -> out_tokens).
        out_blocks = []
        widths = []
        for i, blk in enumerate(blocks):
            if self._downsample_ego or i >= self._num_ego:
                blk = self._shuffle(blk)                       # (B, 256, C) -> (B, out_tokens, C)
            out_blocks.append(blk)
            widths.append(blk.shape[1])                        # out_tokens (downsampled) or 256

        # Project each block to LLM width. Projecting per-block then concatenating is
        # numerically identical to projector(cat(blocks)) — the MLP is per-token, no
        # cross-token mixing — so with both extras off this is the same tensor as before.
        # CCTV blocks optionally use a DISTINCT projector and get a per-camera vector added.
        proj_blocks = []
        cctv_idx = 0
        for i, blk in enumerate(out_blocks):
            is_cctv = i >= self._num_ego
            if is_cctv and self._cctv_projector is not None:
                proj = self._cctv_projector(blk)               # distinct CCTV translator
            else:
                proj = self._projector(blk)                    # shared / ego projector
            if is_cctv and self._cctv_cam_embed is not None:
                proj = proj + self._cctv_cam_embed[cctv_idx]   # (llm_dim,) broadcast over tokens
            if is_cctv:
                cctv_idx += 1
            proj_blocks.append(proj)
        tokens = torch.cat(proj_blocks, dim=1)                 # (B, sum(widths), llm_dim)
        per_block_widths = torch.tensor(widths, dtype=torch.long, device=tokens.device)
        return EncodedVision(
            tokens=tokens,
            per_block_widths=per_block_widths,
            block_present=block_present,
            metadata={},
        )
