"""가드레일 모듈.

민감 정보 마스킹 기능을 제공합니다.
"""

from .sensitive import (
    DetectionPattern,
    DetectionResult,
    SensitiveInfoDetector,
    contains_sensitive_info,
    mask_sensitive_info,
)

__all__ = [
    "DetectionPattern",
    "DetectionResult",
    "SensitiveInfoDetector",
    "contains_sensitive_info",
    "mask_sensitive_info",
]
