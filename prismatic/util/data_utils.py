"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        #print("self.pad_token_id", self.pad_token_id)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )

@dataclass
class PaddedCollatorForActionPrediction_Nav_MMN:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    num_img: int = 1

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        if "lan_prompt" in instances[0]:
            lan_prompts = [instance["lan_prompt"] for instance in instances]
        else:
            lan_prompts = None

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values_goal = [instance["pixel_values_goal"] for instance in instances]
            pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)       
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack actin mask
        action_select_mask = [torch.from_numpy(np.copy(instance["action_select_mask"])) for instance in instances]
        action_select_mask = torch.stack(action_select_mask)

        # Stack goal_pose
        goal_pose = [torch.from_numpy(np.copy(instance["goal_pose"])) for instance in instances]
        goal_pose = torch.stack(goal_pose)

        # Stack obj_pose
        obj_pose_norm = [torch.from_numpy(np.copy(instance["obj_pose_norm"])) for instance in instances]
        obj_pose_norm = torch.stack(obj_pose_norm)

        # Stack cur_image
        cur_image = [torch.from_numpy(np.copy(instance["cur_image"])) for instance in instances]
        cur_image = torch.stack(cur_image)

        # Stack goal_image_8
        goal_image_8 = [torch.from_numpy(np.copy(instance["goal_image_8"])) for instance in instances]
        goal_image_8 = torch.stack(goal_image_8)

        # Stack temp_dist
        temp_dist = [torch.from_numpy(np.copy(instance["temp_dist"])) for instance in instances]
        temp_dist = torch.stack(temp_dist)
        
        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None
        #print(input_ids.size(), labels.size())

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            #attention_mask=attention_mask,
            labels=labels,
            modality_id = [instance["modality_id"] for instance in instances],
            actions=actions,
            action_select_mask=action_select_mask,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            cur_image=cur_image,
            goal_image_8=goal_image_8,       
            temp_dist=temp_dist,     
            img_PIL=[instance["img_PIL"] for instance in instances],
        )
        # gimg_PIL (goal image) is absent in the 4-CCTV / no_goal_image path.
        if "gimg_PIL" in instances[0]:
            output["gimg_PIL"] = [instance["gimg_PIL"] for instance in instances]
        # abs_pose (absolute-pose token input) is opt-in — present iff the
        # dataset's abs_pose flag is on.
        if "abs_pose" in instances[0]:
            output["abs_pose"] = torch.stack(
                [torch.from_numpy(np.copy(instance["abs_pose"])) for instance in instances])
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if lan_prompts is not None:
            output["lan_prompts"] = lan_prompts

        return output


@dataclass
class PaddedCollatorForActionPrediction_Nav_CCTV:
    """CCTV joint-training collate (copy of `_Nav_MMN` + extra image slots).

    Channel-concatenates up to 4 image views in a FIXED order that the model's
    mask builder relies on (ORDER IS LOAD-BEARING):
        [ego_obs, ego_goal, cctv_obs, (cctv_goal)]
    The 4th slot (cctv_goal) is included only when the dataset emits
    `cctv_pixel_values_goal` (i.e. use_cctv_goal=True), so the batch is a 3- or
    4-image tensor. Also stacks the per-sample `block_present` presence vector
    (length n_img + 1) consumed by `_build_multimodal_attention_cctv`.
    """
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    num_img: int = 1

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None
        lan_prompts = [instance["lan_prompt"] for instance in instances] if "lan_prompt" in instances[0] else None

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        if isinstance(pixel_values[0], torch.Tensor):
            # Fixed-order channel concat -> (B, n_img * 6, H, W).
            if "cctv_pixel_values_list" in instances[0]:
                # Multi-CCTV. Slot order: [ego_obs, (ego_goal), cctv_0..N-1].
                # ego_goal is present iff the dataset emits pixel_values_goal
                # (ego-goal + multi-cctv mode); absent in pose-only no_goal_image.
                views = [torch.stack(pixel_values)]                       # slot 0: ego_obs
                if "pixel_values_goal" in instances[0]:                   # slot 1: ego_goal
                    views.append(torch.stack([i["pixel_values_goal"] for i in instances]))
                n_cctv = len(instances[0]["cctv_pixel_values_list"])
                for j in range(n_cctv):
                    views.append(torch.stack([i["cctv_pixel_values_list"][j] for i in instances]))
            else:
                pixel_values_goal = [instance["pixel_values_goal"] for instance in instances]
                views = [
                    torch.stack(pixel_values),        # slot 0: ego_obs
                    torch.stack(pixel_values_goal),   # slot 1: ego_goal
                ]
                if "cctv_pixel_values" in instances[0]:
                    views.append(torch.stack([i["cctv_pixel_values"] for i in instances]))
                if "cctv_pixel_values_goal" in instances[0]:
                    cctv_pixel_values_goal = [instance["cctv_pixel_values_goal"] for instance in instances]
                    views.append(torch.stack(cctv_pixel_values_goal))   # slot 3: cctv_goal
            pixel_values = torch.cat(views, dim=1)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Per-sample block presence: (B, n_img + 1)
        block_present = torch.stack(
            [torch.from_numpy(np.copy(instance["block_present"])) for instance in instances]
        )

        actions            = torch.stack([torch.from_numpy(np.copy(instance["actions"]))            for instance in instances])
        action_select_mask = torch.stack([torch.from_numpy(np.copy(instance["action_select_mask"])) for instance in instances])
        goal_pose          = torch.stack([torch.from_numpy(np.copy(instance["goal_pose"]))          for instance in instances])
        obj_pose_norm      = torch.stack([torch.from_numpy(np.copy(instance["obj_pose_norm"]))      for instance in instances])
        cur_image          = torch.stack([torch.from_numpy(np.copy(instance["cur_image"]))          for instance in instances])
        goal_image_8       = torch.stack([torch.from_numpy(np.copy(instance["goal_image_8"]))       for instance in instances])
        temp_dist          = torch.stack([torch.from_numpy(np.copy(instance["temp_dist"]))          for instance in instances])

        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            labels=labels,
            modality_id=[instance["modality_id"] for instance in instances],
            block_present=block_present,
            actions=actions,
            action_select_mask=action_select_mask,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            cur_image=cur_image,
            goal_image_8=goal_image_8,
            temp_dist=temp_dist,
            img_PIL=[instance["img_PIL"] for instance in instances],
        )
        # gimg_PIL (goal image) is absent in the 4-CCTV / no_goal_image path.
        if "gimg_PIL" in instances[0]:
            output["gimg_PIL"] = [instance["gimg_PIL"] for instance in instances]
        # abs_pose (absolute-pose token input) is opt-in — present iff the
        # dataset's abs_pose flag is on.
        if "abs_pose" in instances[0]:
            output["abs_pose"] = torch.stack(
                [torch.from_numpy(np.copy(instance["abs_pose"])) for instance in instances])
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if lan_prompts is not None:
            output["lan_prompts"] = lan_prompts
        return output


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [instance["pixel_values_goal"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
