"""민감 정보 탐지 모듈.

개인정보 및 민감한 정보를 탐지하고 마스킹합니다.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class DetectionPattern:
    """탐지 패턴 정의.

    Attributes:
        name: 패턴 이름
        pattern: 정규식 패턴
        replacement: 마스킹 대체 문자열
        description: 패턴 설명
    """

    name: str
    pattern: re.Pattern[str]
    replacement: str
    description: str = ""


@dataclass
class DetectionResult:
    """탐지 결과.

    Attributes:
        found: 민감 정보 발견 여부
        patterns_matched: 매칭된 패턴 목록
        masked_text: 마스킹된 텍스트
        original_text: 원본 텍스트
    """

    found: bool
    patterns_matched: list[str] = field(default_factory=list)
    masked_text: str = ""
    original_text: str = ""


class SensitiveInfoDetector:
    """민감 정보 탐지기.

    이메일, 전화번호, 주민번호, API 키 등을 탐지합니다.
    """

    # 기본 탐지 패턴
    DEFAULT_PATTERNS: list[DetectionPattern] = [
        DetectionPattern(
            name="email",
            pattern=re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            replacement="[이메일]",
            description="이메일 주소",
        ),
        DetectionPattern(
            name="phone_kr",
            pattern=re.compile(r"01[0-9]-?\d{3,4}-?\d{4}"),
            replacement="[전화번호]",
            description="한국 휴대폰 번호",
        ),
        DetectionPattern(
            name="phone_landline",
            pattern=re.compile(r"0\d{1,2}-?\d{3,4}-?\d{4}"),
            replacement="[전화번호]",
            description="한국 일반 전화번호",
        ),
        DetectionPattern(
            name="resident_id",
            pattern=re.compile(r"\d{6}-?[1-4]\d{6}"),
            replacement="[주민번호]",
            description="주민등록번호",
        ),
        DetectionPattern(
            name="credit_card",
            pattern=re.compile(r"\d{4}-?\d{4}-?\d{4}-?\d{4}"),
            replacement="[카드번호]",
            description="신용카드 번호",
        ),
        DetectionPattern(
            name="api_key",
            pattern=re.compile(r"(sk-|xoxb-|xapp-|Bearer\s+)[a-zA-Z0-9_-]{20,}"),
            replacement="[API키]",
            description="API 키 또는 토큰",
        ),
        DetectionPattern(
            name="ip_address",
            pattern=re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
            replacement="[IP주소]",
            description="IP 주소",
        ),
        DetectionPattern(
            name="password_pattern",
            pattern=re.compile(
                r"(password|passwd|pwd|비밀번호|패스워드)\s*[:=]\s*\S+",
                re.IGNORECASE,
            ),
            replacement="[비밀번호]",
            description="비밀번호 패턴",
        ),
    ]

    def __init__(
        self,
        patterns: list[DetectionPattern] | None = None,
        custom_filters: list[Callable[[str], str]] | None = None,
    ) -> None:
        """탐지기 초기화.

        Args:
            patterns: 사용할 탐지 패턴 목록 (None이면 기본 패턴 사용)
            custom_filters: 추가 커스텀 필터 함수 목록
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.custom_filters = custom_filters or []

    def detect(self, text: str) -> DetectionResult:
        """민감 정보 탐지.

        Args:
            text: 검사할 텍스트

        Returns:
            탐지 결과
        """
        if not text:
            return DetectionResult(
                found=False,
                masked_text="",
                original_text="",
            )

        patterns_matched: list[str] = []
        masked_text = text

        for pattern in self.patterns:
            if pattern.pattern.search(text):
                patterns_matched.append(pattern.name)
                masked_text = pattern.pattern.sub(pattern.replacement, masked_text)

        # 커스텀 필터 적용
        for filter_func in self.custom_filters:
            masked_text = filter_func(masked_text)

        return DetectionResult(
            found=len(patterns_matched) > 0,
            patterns_matched=patterns_matched,
            masked_text=masked_text,
            original_text=text,
        )

    def contains_sensitive_info(self, text: str) -> bool:
        """민감 정보 포함 여부 확인.

        Args:
            text: 검사할 텍스트

        Returns:
            민감 정보 포함 여부
        """
        return self.detect(text).found

    def mask(self, text: str) -> str:
        """민감 정보 마스킹.

        Args:
            text: 마스킹할 텍스트

        Returns:
            마스킹된 텍스트
        """
        return self.detect(text).masked_text


# 편의 함수
_default_detector = SensitiveInfoDetector()


def mask_sensitive_info(text: str) -> str:
    """민감 정보 마스킹 편의 함수.

    Args:
        text: 마스킹할 텍스트

    Returns:
        마스킹된 텍스트
    """
    return _default_detector.mask(text)


def contains_sensitive_info(text: str) -> bool:
    """민감 정보 포함 여부 확인 편의 함수.

    Args:
        text: 검사할 텍스트

    Returns:
        민감 정보 포함 여부
    """
    return _default_detector.contains_sensitive_info(text)
