"""Configuration for video-processing quality and throughput trade-offs."""

from dataclasses import dataclass
from typing import Literal


ProcessingProfile = Literal["fast", "balanced", "accurate"]


@dataclass(frozen=True)
class VideoProcessingOptions:
    """Inference settings resolved once at the API boundary."""

    profile: ProcessingProfile = "balanced"
    image_size: int = 640
    frame_stride: int = 1


_PROFILES: dict[str, VideoProcessingOptions] = {
    "balanced": VideoProcessingOptions("balanced", image_size=640, frame_stride=1),
    "fast": VideoProcessingOptions("fast", image_size=640, frame_stride=2),
    "accurate": VideoProcessingOptions("accurate", image_size=1280, frame_stride=1),
}


def get_processing_options(profile: ProcessingProfile) -> VideoProcessingOptions:
    """Return immutable settings for a supported profile."""

    return _PROFILES[profile]
