#!/usr/bin/env python3
"""
Training entry point for UmeTrack.
"""

from __future__ import annotations

import argparse
import logging
import os
from functools import partial
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

from lib.common.losses import UmeTrackLoss
from lib.batched_dataset.data_transform import ModelInput, preprocess
from lib.common.hand import HandModel, mirrored_hand_model
from lib.data_utils import bundles
from lib.data_utils.async_dataset import AsyncToIterableDataset, Sampler, find_dataset
from lib.data_utils.dataset_util import map_dataset
from lib.data_utils.split import Split
from lib.models import feature_extractor as fe
from lib.models import skeleton_encoder as se
from lib.models import temporal as tem
from lib.models.model_loader import _create_regressor
from lib.models.model_opts import ModelOpts
from lib.models.regressor import RegressorOutput
from lib.models.umetrack_model import (
    InputFrameData,
    InputFrameDesc,
    InputSkeletonData,
    UmeTrackModel,
)
LOGGER = logging.getLogger("umetrack.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UmeTrack on torch_data")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "UmeTrack_data", "torch_data"),
        help="Path to UmeTrack_data/torch_data",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["real", "synthetic"],
        help="Dataset folders under data-root to include",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--disable-known-regressor",
        action="store_true",
        help="Disable training of the known-skeleton regressor",
    )
    parser.add_argument(
        "--disable-unknown-regressor",
        action="store_true",
        help="Disable training of the unknown-skeleton regressor",
    )
    parser.add_argument(
        "--disable-pose-loss",
        action="store_true",
        help="Disable pose loss; useful for staged training",
    )
    parser.add_argument(
        "--disable-temporal-loss",
        action="store_true",
        help="Disable temporal loss",
    )
    parser.add_argument(
        "--disable-pinch-loss",
        action="store_true",
        help="Disable pinch loss",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--prefetch", type=int, default=64, help="Prefetch queue size")
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="Square crop size (pixels)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max-norm (0 to disable)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint to resume from",
    )
    return parser.parse_args()


def build_model(crop_size: int, device: torch.device) -> UmeTrackModel:
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor((crop_size, crop_size), model_opts)
    temporal_model = tem.create_temporal_model(model_opts, feature_extractor.output_feature_sizes)
    skeleton_encoder = se.SkeletonEncoder(
        [model_opts.nSkeletonFeatureChannels, *feature_extractor.output_feature_sizes]
    )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )
    model = UmeTrackModel(
        feature_extractor=feature_extractor,
        temporal=temporal_model,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )
    return model.to(device)


def _reset_temporal_state(model: UmeTrackModel) -> None:
    if hasattr(model, "_temporal"):
        temporal = model._temporal
        device = temporal._mem_features.device if torch.is_tensor(temporal._mem_features) else torch.device("cpu")
        dtype = temporal._mem_features.dtype if torch.is_tensor(temporal._mem_features) else torch.float32
        temporal._mem_features = torch.empty(0, device=device, dtype=dtype)
        temporal._prev_extrinsics = torch.empty(0, device=device, dtype=torch.float32)


def _prepare_frame_inputs(
    model_input: ModelInput,
    hand_model: HandModel,
    frame_idx: int,
    device: torch.device,
) -> Tuple[InputFrameData, InputFrameDesc, InputSkeletonData | None]:
    left_images = model_input.left_images[:, frame_idx]
    intrinsics = model_input.intrinsics[:, frame_idx]
    extrinsics = model_input.extrinsics_xf[:, frame_idx]
    batch_size, num_views = left_images.shape[:2]

    left_images = left_images.reshape(batch_size * num_views, *left_images.shape[2:]).to(device)
    intrinsics = intrinsics.reshape(batch_size * num_views, *intrinsics.shape[2:]).to(device)
    extrinsics = extrinsics.reshape(batch_size * num_views, *extrinsics.shape[2:]).to(device)

    sample_starts = torch.arange(
        0, batch_size * num_views, num_views, dtype=torch.long, device=device
    )
    sample_range = torch.stack([sample_starts, sample_starts + num_views], dim=-1)

    frame_data = InputFrameData(
        left_images=left_images,
        intrinsics=intrinsics,
        extrinsics_xf=extrinsics,
    )
    frame_desc = InputFrameDesc(
        sample_range=sample_range,
        memory_idx=torch.arange(batch_size, device=device, dtype=torch.long),
        use_memory=torch.full((batch_size,), frame_idx > 0, dtype=torch.bool, device=device),
        hand_idx=model_input.hand_idx[:, frame_idx].long().to(device),
    )

    skeleton_data = InputSkeletonData(
        joint_rotation_axes=hand_model.joint_rotation_axes[:, frame_idx].to(device),
        joint_rest_positions=hand_model.joint_rest_positions[:, frame_idx].to(device),
    )
    return frame_data, frame_desc, skeleton_data


def forward_sequence(
    model: UmeTrackModel,
    model_input: ModelInput,
    hand_model: HandModel,
    device: torch.device,
    mode: str,
) -> RegressorOutput:
    _reset_temporal_state(model)
    seq_len = model_input.left_images.shape[1]
    outputs: List[RegressorOutput] = []

    for frame_idx in range(seq_len):
        frame_data, frame_desc, skel_data = _prepare_frame_inputs(model_input, hand_model, frame_idx, device)
        if mode == "known":
            output = model.regress_pose_use_skeleton(frame_data, frame_desc, skel_data)
        else:
            output = model.regress_pose_pred_skel_scale(frame_data, frame_desc)
        outputs.append(output)

    batched = bundles.collate(outputs)
    batched = bundles.map_fields(
        lambda t: t.transpose(0, 1),
        batched,
        only_type=torch.Tensor,
    )
    return batched


def create_dataloader(
    data_root: str,
    datasets: Iterable[str],
    crop_size: int,
    batch_size: int,
    num_workers: int,
    max_prefetch: int,
) -> DataLoader:
    roots = [os.path.join(data_root, dataset) for dataset in datasets]
    field_names = ["mono", "labels"]
    dataset_map = find_dataset(roots, field_names, splits=[Split.TRAIN])
    train_dataset = dataset_map[Split.TRAIN]
    sampler = Sampler(train_dataset, shuffle=True, drop_last=True, distrib_info=(0, 1))
    iterable = AsyncToIterableDataset(train_dataset, sampler, max_prefetch=max_prefetch)
    iterable = map_dataset(
        partial(preprocess, crop_size=(crop_size, crop_size)),
        iterable,
    )
    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=bundles.collate,
    )
    return loader


def train_one_epoch(
    model: UmeTrackModel,
    dataloader: DataLoader,
    criterion: UmeTrackLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    enable_known: bool,
    enable_unknown: bool,
    grad_clip: float,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = {}
    num_batches = 0

    def accumulate(prefix: str, losses: Dict[str, torch.Tensor]) -> None:
        for name, value in losses.items():
            key = f"{prefix}_{name}"
            running[key] = running.get(key, 0.0) + float(value.detach().cpu())

    for batch_idx, (model_input, model_target) in enumerate(dataloader):
        num_batches += 1
        model_input = bundles.to_device(model_input, device)
        model_target = bundles.to_device(model_target, device)
        hand_model = mirrored_hand_model(
            model_input.orig_pose_data.left_hand_model,
            model_input.hand_idx == 1,
        )
        hand_model = bundles.to_device(hand_model, device)

        optimizer.zero_grad()
        total_loss = None
        log_segments = []

        if enable_known:
            predictions_known = forward_sequence(model, model_input, hand_model, device, mode="known")
            losses_known = criterion(predictions_known, model_target, hand_model)
            total_loss = losses_known["total"] if total_loss is None else total_loss + losses_known["total"]
            accumulate("known", losses_known)
            log_segments.append(
                "K total={:.4f} pose={:.4f} temp={:.4f} pinch={:.4f}".format(
                    losses_known["total"].item(),
                    losses_known["pose"].item(),
                    losses_known["temporal"].item(),
                    losses_known["pinch"].item(),
                )
            )

        has_multi_view = model_input.left_images.shape[2] > 1
        if enable_unknown and has_multi_view:
            predictions_unknown = forward_sequence(model, model_input, hand_model, device, mode="unknown")
            losses_unknown = criterion(predictions_unknown, model_target, hand_model)
            total_loss = losses_unknown["total"] if total_loss is None else total_loss + losses_unknown["total"]
            accumulate("unknown", losses_unknown)
            log_segments.append(
                "U total={:.4f} pose={:.4f} temp={:.4f} pinch={:.4f}".format(
                    losses_unknown["total"].item(),
                    losses_unknown["pose"].item(),
                    losses_unknown["temporal"].item(),
                    losses_unknown["pinch"].item(),
                )
            )
        elif enable_unknown and not has_multi_view and batch_idx == 0:
            LOGGER.warning("Skipping unknown regressor for single-view batch")

        if total_loss is None:
            continue

        total_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if batch_idx % 10 == 0 and log_segments:
            LOGGER.info("Epoch %d Batch %d: %s", epoch, batch_idx, " | ".join(log_segments))

    num_batches = max(1, num_batches)
    return {k: v / num_batches for k, v in running.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    dataloader = create_dataloader(
        data_root=args.data_root,
        datasets=args.datasets,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_prefetch=args.prefetch,
    )

    model = build_model(args.crop_size, device)
    if args.resume:
        LOGGER.info("Loading checkpoint %s", args.resume)
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    enable_known = not args.disable_known_regressor
    enable_unknown = not args.disable_unknown_regressor
    if not enable_known and not enable_unknown:
        raise ValueError("At least one of known/unknown regressors must be enabled.")

    criterion = UmeTrackLoss(
        enable_pose=not args.disable_pose_loss,
        enable_temporal=not args.disable_temporal_loss,
        enable_pinch=not args.disable_pinch_loss,
    )

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            enable_known=enable_known,
            enable_unknown=enable_unknown,
            grad_clip=args.grad_clip,
            epoch=epoch,
        )
        LOGGER.info("Epoch %d metrics: %s", epoch, metrics)

        ckpt_path = os.path.join(args.output_dir, f"umetrack_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        LOGGER.info("Saved checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()

