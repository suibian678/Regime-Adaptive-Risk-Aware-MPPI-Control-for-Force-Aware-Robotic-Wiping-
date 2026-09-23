"""Parameter-matched modality encoders for the V4 observation study."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .decision_protocol import DecisionProtocolError
from .information_flow import ObservationMode
from .modality_adapter import FORCE_HISTORY_FEATURES, MatchedModalityPacket


@dataclass(frozen=True)
class MatchedPolicyEncoderConfig:
    residual_cell_count: int = 64
    force_history_length: int = 32
    force_history_feature_count: int = len(FORCE_HISTORY_FEATURES)
    common_core_dimension: int = 18
    embedding_dimension: int = 128
    vision_hidden_dimension: int = 96
    force_hidden_dimension: int = 53
    fusion_hidden_dimension: int = 43
    maximum_parameter_spread_fraction: float = 0.05

    def validate(self) -> None:
        dimensions = (
            self.residual_cell_count,
            self.force_history_length,
            self.force_history_feature_count,
            self.common_core_dimension,
            self.embedding_dimension,
            self.vision_hidden_dimension,
            self.force_hidden_dimension,
            self.fusion_hidden_dimension,
        )
        if not all(int(value) > 0 for value in dimensions):
            raise DecisionProtocolError("matched encoder dimensions must be positive")
        spread = float(self.maximum_parameter_spread_fraction)
        if not math.isfinite(spread) or not 0 <= spread < 1:
            raise DecisionProtocolError("parameter spread fraction must lie in [0, 1)")


class _TwoLayerEncoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dimension), int(hidden_dimension)),
            nn.SiLU(),
            nn.Linear(int(hidden_dimension), int(output_dimension)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class MatchedPolicyEncoder(nn.Module):
    """One modality-specific adapter with a shared output dimension."""

    def __init__(
        self,
        mode: ObservationMode,
        config: MatchedPolicyEncoderConfig = MatchedPolicyEncoderConfig(),
    ) -> None:
        super().__init__()
        config.validate()
        self.mode = ObservationMode(mode)
        if self.mode not in {
            ObservationMode.VISION_ONLY,
            ObservationMode.FORCE_ONLY,
            ObservationMode.FUSION,
        }:
            raise DecisionProtocolError("matched policy encoder supports three nonoracle modalities")
        self.config = config
        visual_dimension = 2 * int(config.residual_cell_count)
        force_dimension = int(config.force_history_length) * (
            int(config.force_history_feature_count) + 1
        )
        if self.mode == ObservationMode.VISION_ONLY:
            input_dimension = visual_dimension
            hidden_dimension = int(config.vision_hidden_dimension)
        elif self.mode == ObservationMode.FORCE_ONLY:
            input_dimension = force_dimension
            hidden_dimension = int(config.force_hidden_dimension)
        else:
            input_dimension = visual_dimension + force_dimension
            hidden_dimension = int(config.fusion_hidden_dimension)
        self.modality_encoder = _TwoLayerEncoder(
            input_dimension,
            hidden_dimension,
            int(config.embedding_dimension),
        )

    def forward(
        self,
        common_core: torch.Tensor,
        *,
        visual_features: torch.Tensor | None = None,
        force_history: torch.Tensor | None = None,
        force_history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if common_core.ndim != 2 or common_core.shape[1] != self.config.common_core_dimension:
            raise DecisionProtocolError("common policy core has the wrong shape")
        batch = common_core.shape[0]
        visual_required = self.mode in {ObservationMode.VISION_ONLY, ObservationMode.FUSION}
        force_required = self.mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}
        if visual_required:
            expected_visual = 2 * int(self.config.residual_cell_count)
            if visual_features is None or visual_features.shape != (batch, expected_visual):
                raise DecisionProtocolError("visual modality features have the wrong shape")
        elif visual_features is not None:
            raise DecisionProtocolError("force-only encoder received visual features")
        if force_required:
            expected_history = (
                batch,
                int(self.config.force_history_length),
                int(self.config.force_history_feature_count),
            )
            expected_mask = (batch, int(self.config.force_history_length))
            if force_history is None or force_history.shape != expected_history:
                raise DecisionProtocolError("force history has the wrong shape")
            if force_history_mask is None or force_history_mask.shape != expected_mask:
                raise DecisionProtocolError("force-history mask has the wrong shape")
            masked_history = force_history * force_history_mask.unsqueeze(-1)
            force_features = torch.cat(
                (masked_history.reshape(batch, -1), force_history_mask), dim=1
            )
        elif force_history is not None or force_history_mask is not None:
            raise DecisionProtocolError("vision-only encoder received force-history features")
        if self.mode == ObservationMode.VISION_ONLY:
            modality_input = visual_features
        elif self.mode == ObservationMode.FORCE_ONLY:
            modality_input = force_features
        else:
            modality_input = torch.cat((visual_features, force_features), dim=1)
        embedding = self.modality_encoder(modality_input)
        return torch.cat((common_core, embedding), dim=1)

    @torch.no_grad()
    def encode_packet(
        self,
        packet: MatchedModalityPacket,
        *,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if ObservationMode(packet.mode) != self.mode or packet.privileged_oracle:
            raise DecisionProtocolError("modality packet does not match the encoder")
        if (
            int(packet.force_history_length) != self.config.force_history_length
            or int(packet.force_history_feature_count)
            != self.config.force_history_feature_count
        ):
            raise DecisionProtocolError("packet history metadata differs from encoder contract")
        decision = np.asarray(packet.decision_vector, dtype=np.float32)
        expected_decision = self.config.common_core_dimension + 2 * self.config.residual_cell_count
        if decision.shape != (expected_decision,) or not np.all(np.isfinite(decision)):
            raise DecisionProtocolError("packet decision vector differs from encoder contract")
        common = torch.from_numpy(decision[: self.config.common_core_dimension]).unsqueeze(0).to(device)
        visual = None
        if self.mode in {ObservationMode.VISION_ONLY, ObservationMode.FUSION}:
            visual = torch.from_numpy(
                decision[self.config.common_core_dimension :].copy()
            ).unsqueeze(0).to(device)
        history = None
        mask = None
        if self.mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}:
            history_array = np.asarray(packet.force_history_flat, dtype=np.float32)
            mask_array = np.asarray(packet.force_history_mask, dtype=np.float32)
            expected_history = (
                self.config.force_history_length * self.config.force_history_feature_count
            )
            if history_array.shape != (expected_history,) or mask_array.shape != (
                self.config.force_history_length,
            ):
                raise DecisionProtocolError("packet force history differs from encoder contract")
            if not np.all(np.isfinite(history_array)) or not np.all(np.isfinite(mask_array)):
                raise DecisionProtocolError("packet force history contains NaN/Inf")
            if not np.all((mask_array == 0.0) | (mask_array == 1.0)):
                raise DecisionProtocolError("packet force-history mask must be binary")
            history = torch.from_numpy(history_array.copy()).reshape(
                1,
                self.config.force_history_length,
                self.config.force_history_feature_count,
            ).to(device)
            mask = torch.from_numpy(mask_array.copy()).unsqueeze(0).to(device)
        return self.forward(
            common,
            visual_features=visual,
            force_history=history,
            force_history_mask=mask,
        )


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def matched_parameter_audit(
    config: MatchedPolicyEncoderConfig = MatchedPolicyEncoderConfig(),
) -> dict:
    config.validate()
    counts = {
        mode.value: trainable_parameter_count(MatchedPolicyEncoder(mode, config))
        for mode in (
            ObservationMode.VISION_ONLY,
            ObservationMode.FORCE_ONLY,
            ObservationMode.FUSION,
        )
    }
    mean_count = float(np.mean(list(counts.values())))
    spread = float((max(counts.values()) - min(counts.values())) / mean_count)
    return {
        "trainable_parameters": counts,
        "mean_trainable_parameters": mean_count,
        "spread_fraction": spread,
        "maximum_spread_fraction": float(config.maximum_parameter_spread_fraction),
        "pass": bool(spread <= config.maximum_parameter_spread_fraction),
        "frozen_visual_cnn_parameters_excluded": True,
        "common_downstream_policy_parameters_excluded_because_identical": True,
    }
