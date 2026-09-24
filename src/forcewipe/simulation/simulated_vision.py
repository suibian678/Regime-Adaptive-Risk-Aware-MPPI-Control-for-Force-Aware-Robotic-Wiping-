"""Simulated RGB sensing for the ForceWipe V4 residual strip.

This module deliberately separates environment-side image formation from the
image-only decoder.  The decoder receives an immutable RGB frame and camera
calibration; it has no residual-state or audit capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from forcewipe.simulation.residual import ResidualContractError


@dataclass(frozen=True)
class ResidualStripCameraConfig:
    """Calibration and rendering parameters for a top-down residual strip."""

    image_height_px: int = 24
    pixels_per_cell: int = 4
    clean_rgb: tuple[int, int, int] = (214, 222, 226)
    residual_rgb: tuple[int, int, int] = (112, 70, 38)
    illumination_std: float = 0.0
    white_balance_std: float = 0.0
    horizontal_shift_max_px: int = 0
    occlusion_probability: float = 0.0
    occlusion_max_area_fraction: float = 0.0
    occluder_rgb: tuple[int, int, int] = (54, 58, 62)
    blur_radius_px: int = 0

    def validate(self) -> None:
        if int(self.image_height_px) <= 0 or int(self.pixels_per_cell) <= 0:
            raise ResidualContractError("camera dimensions must be positive")
        for name, colour in (
            ("clean_rgb", self.clean_rgb),
            ("residual_rgb", self.residual_rgb),
            ("occluder_rgb", self.occluder_rgb),
        ):
            if len(colour) != 3 or any(int(value) != value or not 0 <= value <= 255 for value in colour):
                raise ResidualContractError(f"{name} must contain three RGB8 values")
        if self.clean_rgb == self.residual_rgb:
            raise ResidualContractError("clean and residual colours must differ")
        if not math.isfinite(float(self.illumination_std)) or self.illumination_std < 0:
            raise ResidualContractError("illumination_std must be finite and nonnegative")
        if not math.isfinite(float(self.white_balance_std)) or self.white_balance_std < 0:
            raise ResidualContractError("white_balance_std must be finite and nonnegative")
        if int(self.horizontal_shift_max_px) != self.horizontal_shift_max_px or self.horizontal_shift_max_px < 0:
            raise ResidualContractError("horizontal_shift_max_px must be a nonnegative integer")
        probability_fields = (
            self.occlusion_probability,
            self.occlusion_max_area_fraction,
        )
        if not all(math.isfinite(float(value)) and 0 <= value <= 1 for value in probability_fields):
            raise ResidualContractError("occlusion probability/area must lie in [0, 1]")
        if int(self.blur_radius_px) != self.blur_radius_px or self.blur_radius_px < 0:
            raise ResidualContractError("blur_radius_px must be a nonnegative integer")


@dataclass(frozen=True)
class SimulatedRGBFrame:
    """Immutable RGB8 observation packet with no privileged residual fields."""

    sample_index: int
    time_ns: int
    height_px: int
    width_px: int
    cell_count: int
    pixels_rgb8: bytes

    def __post_init__(self) -> None:
        if int(self.sample_index) < 0 or int(self.time_ns) < 0:
            raise ResidualContractError("RGB frame index/time must be nonnegative")
        if int(self.height_px) <= 0 or int(self.width_px) <= 0 or int(self.cell_count) <= 0:
            raise ResidualContractError("RGB frame dimensions must be positive")
        payload = bytes(self.pixels_rgb8)
        expected = int(self.height_px) * int(self.width_px) * 3
        if len(payload) != expected:
            raise ResidualContractError(
                f"RGB payload has {len(payload)} bytes; expected {expected}"
            )
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "time_ns", int(self.time_ns))
        object.__setattr__(self, "height_px", int(self.height_px))
        object.__setattr__(self, "width_px", int(self.width_px))
        object.__setattr__(self, "cell_count", int(self.cell_count))
        object.__setattr__(self, "pixels_rgb8", payload)

    def as_array(self) -> np.ndarray:
        """Return a read-only ``H x W x 3`` view of the immutable payload."""

        image = np.frombuffer(self.pixels_rgb8, dtype=np.uint8).reshape(
            self.height_px, self.width_px, 3
        )
        image.setflags(write=False)
        return image


def render_residual_strip_rgb(
    residual_state: np.ndarray,
    *,
    sample_index: int,
    time_ns: int,
    config: ResidualStripCameraConfig,
    rng: np.random.Generator,
    residual_bias: float = 0.0,
    residual_noise_std: float = 0.0,
) -> SimulatedRGBFrame:
    """Environment-side image formation from the hidden residual state."""

    config.validate()
    residual = np.asarray(residual_state, dtype=np.float64)
    if residual.ndim != 1 or residual.size == 0:
        raise ResidualContractError("residual image source must be a nonempty vector")
    if not np.all(np.isfinite(residual)) or np.any(residual < 0) or np.any(residual > 1):
        raise ResidualContractError("residual image source must be finite and in [0, 1]")
    if not math.isfinite(float(residual_bias)) or not math.isfinite(float(residual_noise_std)):
        raise ResidualContractError("visual bias/noise must be finite")
    if residual_noise_std < 0:
        raise ResidualContractError("visual noise must be nonnegative")

    height = int(config.image_height_px)
    pixels_per_cell = int(config.pixels_per_cell)
    alpha = np.broadcast_to(
        residual.reshape(1, residual.size, 1),
        (height, residual.size, pixels_per_cell),
    ).copy()
    if residual_noise_std > 0 or residual_bias != 0:
        alpha += rng.normal(
            loc=float(residual_bias),
            scale=float(residual_noise_std),
            size=alpha.shape,
        )
    alpha = np.clip(alpha, 0.0, 1.0).reshape(height, residual.size * pixels_per_cell, 1)

    clean = np.asarray(config.clean_rgb, dtype=np.float64).reshape(1, 1, 3)
    residual_colour = np.asarray(config.residual_rgb, dtype=np.float64).reshape(1, 1, 3)
    image = clean + alpha * (residual_colour - clean)
    if config.illumination_std > 0:
        illumination = rng.normal(1.0, config.illumination_std, size=(height, 1, 1))
        image *= illumination
    if config.white_balance_std > 0:
        white_balance = rng.normal(1.0, config.white_balance_std, size=(1, 1, 3))
        image *= white_balance

    shift_limit = int(config.horizontal_shift_max_px)
    if shift_limit > 0:
        shift = int(rng.integers(-shift_limit, shift_limit + 1))
        if shift != 0:
            shifted = np.broadcast_to(clean, image.shape).copy()
            if shift > 0:
                shifted[:, shift:, :] = image[:, :-shift, :]
            else:
                shifted[:, :shift, :] = image[:, -shift:, :]
            image = shifted

    if (
        config.occlusion_probability > 0
        and config.occlusion_max_area_fraction > 0
        and rng.random() < config.occlusion_probability
    ):
        width = image.shape[1]
        area_fraction = float(
            rng.uniform(0.25 * config.occlusion_max_area_fraction, config.occlusion_max_area_fraction)
        )
        occlusion_height = int(rng.integers(max(1, height // 4), height + 1))
        occlusion_width = max(
            1,
            min(width, int(round(area_fraction * height * width / occlusion_height))),
        )
        top = int(rng.integers(0, height - occlusion_height + 1))
        left = int(rng.integers(0, width - occlusion_width + 1))
        image[
            top : top + occlusion_height,
            left : left + occlusion_width,
            :,
        ] = np.asarray(config.occluder_rgb, dtype=np.float64)

    radius = int(config.blur_radius_px)
    if radius > 0:
        padded = np.pad(image, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        integral = np.cumsum(padded, axis=1, dtype=np.float64)
        integral = np.concatenate(
            (np.zeros((height, 1, 3), dtype=np.float64), integral), axis=1
        )
        window = 2 * radius + 1
        image = (integral[:, window:, :] - integral[:, :-window, :]) / window
    image_rgb8 = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return SimulatedRGBFrame(
        sample_index=sample_index,
        time_ns=time_ns,
        height_px=height,
        width_px=residual.size * pixels_per_cell,
        cell_count=int(residual.size),
        pixels_rgb8=image_rgb8.tobytes(order="C"),
    )


def decode_residual_strip_rgb(
    frame: SimulatedRGBFrame,
    *,
    config: ResidualStripCameraConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Decode per-cell residual and uncertainty from RGB pixels only."""

    config.validate()
    expected_width = frame.cell_count * int(config.pixels_per_cell)
    if frame.height_px != int(config.image_height_px) or frame.width_px != expected_width:
        raise ResidualContractError("RGB frame geometry differs from camera calibration")

    image = frame.as_array().astype(np.float64)
    clean = np.asarray(config.clean_rgb, dtype=np.float64)
    direction = np.asarray(config.residual_rgb, dtype=np.float64) - clean
    direction_energy = float(np.dot(direction, direction))
    projected = np.tensordot(image - clean, direction, axes=([2], [0])) / direction_energy
    projected = np.clip(projected, 0.0, 1.0).reshape(
        frame.height_px, frame.cell_count, int(config.pixels_per_cell)
    )
    samples = np.transpose(projected, (1, 0, 2)).reshape(frame.cell_count, -1)
    estimate = np.mean(samples, axis=1)
    sample_variance = np.var(samples, axis=1, ddof=1) if samples.shape[1] > 1 else np.zeros(frame.cell_count)
    quantization_variance = 3.0 * (0.5**2) / direction_energy
    mean_variance = sample_variance / max(samples.shape[1], 1) + quantization_variance
    return (
        tuple(float(value) for value in np.clip(estimate, 0.0, 1.0)),
        tuple(float(max(value, 1e-12)) for value in mean_variance),
    )
