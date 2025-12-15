"""리액션 유틸리티.

Slack 메시지에 리액션을 추가/제거하는 기능을 제공합니다.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def add_reaction_safe(
    client: Any,
    channel: str,
    timestamp: str,
    name: str,
) -> bool:
    """메시지에 리액션 추가 (실패 시 무시).

    리액션 실패가 핵심 기능에 영향을 주지 않도록 합니다.

    Args:
        client: Slack WebClient
        channel: 채널 ID
        timestamp: 메시지 타임스탬프
        name: 리액션 이름 (콜론 제외, e.g., "eyes", "white_check_mark")

    Returns:
        성공 여부
    """
    try:
        client.reactions_add(
            channel=channel,
            timestamp=timestamp,
            name=name,
        )
        logger.debug(f"리액션 추가: {name} -> {channel}:{timestamp}")
        return True
    except Exception as e:
        error_str = str(e)
        # 이미 리액션이 있는 경우는 성공으로 처리
        if "already_reacted" in error_str:
            logger.debug(f"리액션 이미 존재: {name} -> {channel}:{timestamp}")
            return True
        logger.warning(f"리액션 추가 실패: {name} -> {channel}:{timestamp}: {e}")
        return False


def remove_reaction_safe(
    client: Any,
    channel: str,
    timestamp: str,
    name: str,
) -> bool:
    """메시지에서 리액션 제거 (실패 시 무시).

    Args:
        client: Slack WebClient
        channel: 채널 ID
        timestamp: 메시지 타임스탬프
        name: 리액션 이름

    Returns:
        성공 여부
    """
    try:
        client.reactions_remove(
            channel=channel,
            timestamp=timestamp,
            name=name,
        )
        logger.debug(f"리액션 제거: {name} -> {channel}:{timestamp}")
        return True
    except Exception as e:
        error_str = str(e)
        # 리액션이 없는 경우는 성공으로 처리
        if "no_reaction" in error_str:
            logger.debug(f"리액션 없음: {name} -> {channel}:{timestamp}")
            return True
        logger.warning(f"리액션 제거 실패: {name} -> {channel}:{timestamp}: {e}")
        return False
