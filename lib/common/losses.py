"""
Training losses for UmeTrack.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.batched_dataset.data_transform import ModelTarget
from lib.common.hand_skinning import skin_landmarks
from lib.models.regressor import RegressorOutput


THUMB_LANDMARK_INDEX = 0
INDEX_LANDMARK_INDEX = 1


def _second_order_difference(values: torch.Tensor) -> torch.Tensor:
    """
    Compute discrete second derivative along the sequence dimension (dim=1).
    """
    if values.shape[1] < 3:
        return torch.zeros_like(values[:, :1])
    return values[:, 2:] + values[:, :-2] - 2 * values[:, 1:-1]


class UmeTrackLoss(nn.Module):
    """
    Composite loss described in the UmeTrack paper (pose + temporal + pinch).
    """

    def __init__(
        self,
        lambda_theta: float = 0.05,
        lambda_wrist: float = 0.5,
        lambda_temporal: float = 0.05,
        lambda_pinch: float = 0.4,
        pinch_close_threshold: float = 0.01,
        pinch_far_threshold: float = 0.02,
        enable_pose: bool = True,
        enable_temporal: bool = True,
        enable_pinch: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_theta = lambda_theta
        self.lambda_wrist = lambda_wrist
        self.lambda_temporal = lambda_temporal
        self.lambda_pinch = lambda_pinch
        self.pinch_close_threshold = pinch_close_threshold
        self.pinch_far_threshold = pinch_far_threshold
        self.enable_pose = enable_pose
        self.enable_temporal = enable_temporal
        self.enable_pinch = enable_pinch

    def forward(
        self,
        predictions: RegressorOutput,
        targets: ModelTarget,
        hand_model,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss terms.
        """
        pose_loss, pose_terms = self._pose_loss(predictions, targets, hand_model)
        temporal_loss = self._temporal_loss(predictions)
        pinch_loss = self._pinch_loss(predictions, targets, hand_model)

        total = pose_loss + self.lambda_temporal * temporal_loss + self.lambda_pinch * pinch_loss

        return {
            "total": total,
            "pose": pose_loss,
            "temporal": temporal_loss,
            "pinch": pinch_loss,
            **pose_terms,
        }

    def _pose_loss(
        self,
        predictions: RegressorOutput,
        targets: ModelTarget,
        hand_model,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enable_pose:
            zero = torch.zeros(
                1, device=predictions.joint_angles.device, dtype=predictions.joint_angles.dtype
            ).squeeze()
            return zero, {"pose_keypoint": zero, "pose_joint": zero, "pose_wrist": zero}

        gt_targets = targets.gt_skel_targets

        pred_keypoints = skin_landmarks(hand_model, predictions.joint_angles, predictions.wrist_xfs)
        gt_keypoints = skin_landmarks(hand_model, gt_targets.joint_angles, gt_targets.wrist_xfs)
        keypoint_loss = F.l1_loss(pred_keypoints, gt_keypoints)

        joint_loss = F.l1_loss(predictions.joint_angles, gt_targets.joint_angles)

        pred_wrist = predictions.wrist_xfs[..., :3, 3]
        gt_wrist = gt_targets.wrist_xfs[..., :3, 3]
        wrist_loss = F.l1_loss(pred_wrist, gt_wrist)

        pose_loss = keypoint_loss + self.lambda_theta * joint_loss + self.lambda_wrist * wrist_loss
        pose_terms = {
            "pose_keypoint": keypoint_loss,
            "pose_joint": joint_loss,
            "pose_wrist": wrist_loss,
        }
        return pose_loss, pose_terms

    def _temporal_loss(self, predictions: RegressorOutput) -> torch.Tensor:
        if not self.enable_temporal:
            return torch.zeros(
                1, device=predictions.joint_angles.device, dtype=predictions.joint_angles.dtype
            ).squeeze()

        if predictions.joint_angles.shape[1] < 3:
            return torch.zeros(
                1, device=predictions.joint_angles.device, dtype=predictions.joint_angles.dtype
            ).squeeze()

        joint_acc = _second_order_difference(predictions.joint_angles)
        wrist_acc = _second_order_difference(predictions.wrist_xfs[..., :3, 3])

        joint_term = joint_acc.abs().mean()
        wrist_term = wrist_acc.abs().mean()
        return joint_term + wrist_term

    def _pinch_loss(
        self,
        predictions: RegressorOutput,
        targets: ModelTarget,
        hand_model,
    ) -> torch.Tensor:
        if not self.enable_pinch:
            return torch.zeros(
                1, device=predictions.joint_angles.device, dtype=predictions.joint_angles.dtype
            ).squeeze()

        pinch_labels = targets.gt_skel_targets.pinch_prediction
        if pinch_labels is None:
            return torch.zeros(
                1, device=predictions.joint_angles.device, dtype=predictions.joint_angles.dtype
            ).squeeze()

        pinch_labels = pinch_labels.float()
        if pinch_labels.dim() == 3:
            pinch_labels = pinch_labels.squeeze(-1)

        pred_keypoints = skin_landmarks(hand_model, predictions.joint_angles, predictions.wrist_xfs)
        thumb_pts = pred_keypoints[:, :, THUMB_LANDMARK_INDEX]
        index_pts = pred_keypoints[:, :, INDEX_LANDMARK_INDEX]
        pinch_distance = torch.linalg.norm(thumb_pts - index_pts, dim=-1)

        pinch_active = pinch_labels
        pinch_inactive = 1.0 - pinch_active

        close_violation = F.relu(pinch_distance - self.pinch_close_threshold)
        open_violation = F.relu(self.pinch_far_threshold - pinch_distance)
        pinch_loss = (pinch_active * close_violation + pinch_inactive * open_violation).mean()
        return pinch_loss

