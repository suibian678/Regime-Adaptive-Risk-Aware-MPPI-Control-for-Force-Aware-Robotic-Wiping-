"""RGB-only convolutional residual encoder for ForceWipe V4 method studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .residual import ResidualContractError
from .simulated_vision import ResidualStripCameraConfig, SimulatedRGBFrame


@dataclass(frozen=True)
class ResidualCNNConfig:
    cell_count: int = 64
    base_channels: int = 16
    decoder_variance: float = 0.01

    def validate(self) -> None:
        if int(self.cell_count) <= 0 or int(self.base_channels) <= 0:
            raise ResidualContractError("visual encoder dimensions must be positive")
        if not math.isfinite(float(self.decoder_variance)) or self.decoder_variance <= 0:
            raise ResidualContractError("decoder variance must be finite and positive")


class ResidualStripCNN(nn.Module):
    """Fully convolutional RGB-to-residual model with no privileged input port."""

    def __init__(self, config: ResidualCNNConfig = ResidualCNNConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        channels = int(config.base_channels)
        self.image_features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                2 * channels,
                kernel_size=(3, 5),
                stride=(2, 2),
                padding=(1, 2),
            ),
            nn.SiLU(),
            nn.Conv2d(
                2 * channels,
                4 * channels,
                kernel_size=3,
                stride=(2, 2),
                padding=1,
            ),
            nn.SiLU(),
        )
        self.spatial_decoder = nn.Sequential(
            nn.Conv1d(4 * channels, 2 * channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(2 * channels, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ResidualContractError("visual encoder expects Bx3xHxW RGB tensors")
        features = torch.mean(self.image_features(images), dim=2)
        if features.shape[-1] != int(self.config.cell_count):
            features = nn.functional.interpolate(
                features,
                size=int(self.config.cell_count),
                mode="linear",
                align_corners=False,
            )
        logits = self.spatial_decoder(features).squeeze(1)
        return torch.sigmoid(logits)


def frame_to_tensor(
    frame: SimulatedRGBFrame,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    image = frame.as_array().copy()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    return tensor / 255.0


class TorchVisualResidualDecoder:
    """Callable image-only decoder compatible with the visual sensor boundary."""

    def __init__(
        self,
        model: ResidualStripCNN,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(
        self,
        frame: SimulatedRGBFrame,
        camera_config: ResidualStripCameraConfig,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        camera_config.validate()
        if frame.cell_count != int(self.model.config.cell_count):
            raise ResidualContractError("RGB frame cell count differs from encoder output")
        prediction = self.model(frame_to_tensor(frame, device=self.device))[0]
        values = prediction.detach().cpu().to(torch.float64).numpy()
        if values.shape != (frame.cell_count,) or not np.all(np.isfinite(values)):
            raise ResidualContractError("visual encoder returned an invalid residual vector")
        variance = float(self.model.config.decoder_variance)
        return (
            tuple(float(value) for value in np.clip(values, 0.0, 1.0)),
            tuple(variance for _ in range(frame.cell_count)),
        )


def save_visual_encoder_checkpoint(
    path: Path,
    model: ResidualStripCNN,
    *,
    metadata: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "forcewipe_v4_residual_cnn_v1",
        "config": asdict(model.config),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }
    torch.save(payload, destination)


def load_visual_encoder_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ResidualStripCNN, Mapping[str, Any]]:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    except Exception as error:
        raise ResidualContractError("visual encoder checkpoint could not be loaded safely") from error
    if not isinstance(payload, dict) or payload.get("format") != "forcewipe_v4_residual_cnn_v1":
        raise ResidualContractError("visual encoder checkpoint format is invalid")
    try:
        config = ResidualCNNConfig(**payload["config"])
        model = ResidualStripCNN(config)
        model.load_state_dict(payload["state_dict"], strict=True)
        metadata = payload["metadata"]
    except Exception as error:
        raise ResidualContractError("visual encoder checkpoint contents are invalid") from error
    if not isinstance(metadata, Mapping):
        raise ResidualContractError("visual encoder checkpoint metadata is invalid")
    model.to(device).eval()
    return model, metadata
