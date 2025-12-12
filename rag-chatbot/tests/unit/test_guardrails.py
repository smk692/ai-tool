"""가드레일 모듈 테스트."""


from src.guardrails import (
    SensitiveInfoDetector,
    contains_sensitive_info,
    mask_sensitive_info,
)


class TestSensitiveInfoDetector:
    """SensitiveInfoDetector 클래스 테스트."""

    def test_detect_email(self) -> None:
        """이메일 탐지 테스트."""
        detector = SensitiveInfoDetector()
        result = detector.detect("연락처: test@example.com 입니다")

        assert result.found is True
        assert "email" in result.patterns_matched
        assert "[이메일]" in result.masked_text

    def test_detect_phone_kr(self) -> None:
        """한국 휴대폰 번호 탐지 테스트."""
        detector = SensitiveInfoDetector()
        result = detector.detect("전화번호는 010-1234-5678 입니다")

        assert result.found is True
        assert "phone_kr" in result.patterns_matched
        assert "[전화번호]" in result.masked_text

    def test_detect_resident_id(self) -> None:
        """주민등록번호 탐지 테스트."""
        detector = SensitiveInfoDetector()
        # 주민번호 뒷자리가 전화번호 패턴과 충돌하지 않도록
        result = detector.detect("주민번호: 850315-2876543")

        assert result.found is True
        assert "resident_id" in result.patterns_matched
        # 주민번호 패턴 탐지 여부만 확인 (다른 패턴과 중복 가능)

    def test_detect_credit_card(self) -> None:
        """신용카드 번호 탐지 테스트."""
        detector = SensitiveInfoDetector()
        result = detector.detect("카드번호 1234-5678-9012-3456")

        assert result.found is True
        assert "credit_card" in result.patterns_matched
        assert "[카드번호]" in result.masked_text

    def test_detect_api_key(self) -> None:
        """API 키 탐지 테스트."""
        detector = SensitiveInfoDetector()
        result = detector.detect("API Key: sk-abcdefghijklmnopqrstuvwxyz")

        assert result.found is True
        assert "api_key" in result.patterns_matched
        assert "[API키]" in result.masked_text

    def test_detect_ip_address(self) -> None:
        """IP 주소 탐지 테스트."""
        detector = SensitiveInfoDetector()
        result = detector.detect("서버 IP: 192.168.1.100")

        assert result.found is True
        assert "ip_address" in result.patterns_matched
        assert "[IP주소]" in result.masked_text

    def test_detect_password_pattern(self) -> None:
        """비밀번호 패턴 탐지 테스트."""
        detector = SensitiveInfoDetector()
        result = detector.detect("password: mySecretPass123")

        assert result.found is True
        assert "password_pattern" in result.patterns_matched
        assert "[비밀번호]" in result.masked_text

    def test_no_sensitive_info(self) -> None:
        """민감 정보 없음 테스트."""
        detector = SensitiveInfoDetector()
        result = detector.detect("이것은 일반 텍스트입니다.")

        assert result.found is False
        assert len(result.patterns_matched) == 0
        assert result.masked_text == "이것은 일반 텍스트입니다."

    def test_empty_text(self) -> None:
        """빈 텍스트 테스트."""
        detector = SensitiveInfoDetector()
        result = detector.detect("")

        assert result.found is False
        assert result.masked_text == ""

    def test_multiple_sensitive_info(self) -> None:
        """여러 민감 정보 탐지 테스트."""
        detector = SensitiveInfoDetector()
        text = "이메일: test@example.com, 전화: 010-1234-5678"
        result = detector.detect(text)

        assert result.found is True
        assert "email" in result.patterns_matched
        assert "phone_kr" in result.patterns_matched
        assert "[이메일]" in result.masked_text
        assert "[전화번호]" in result.masked_text


class TestConvenienceFunctions:
    """편의 함수 테스트."""

    def test_mask_sensitive_info(self) -> None:
        """마스킹 편의 함수 테스트."""
        text = "연락처: test@example.com"
        masked = mask_sensitive_info(text)

        assert "[이메일]" in masked
        assert "test@example.com" not in masked

    def test_contains_sensitive_info_true(self) -> None:
        """민감 정보 포함 확인 테스트 (True)."""
        result = contains_sensitive_info("전화: 010-1234-5678")
        assert result is True

    def test_contains_sensitive_info_false(self) -> None:
        """민감 정보 포함 확인 테스트 (False)."""
        result = contains_sensitive_info("일반 텍스트입니다")
        assert result is False
