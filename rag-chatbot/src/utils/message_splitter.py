"""Slack 메시지 분할 유틸리티.

긴 메시지를 Slack 제한에 맞게 분할하며, 코드 블록 경계를 안전하게 처리합니다.
"""

import re

# Slack 메시지 제한 (안전 마진 포함)
SLACK_MESSAGE_LIMIT = 3900  # 실제 제한 ~4000, 여유분 확보
SLACK_BLOCK_TEXT_LIMIT = 2900  # Block Kit text 제한 ~3000


def split_message(
    text: str,
    max_length: int = SLACK_MESSAGE_LIMIT,
) -> list[str]:
    """긴 메시지를 여러 청크로 분할.

    코드 블록(```)을 안전하게 처리하여 중간에 끊기지 않도록 합니다.

    Args:
        text: 분할할 메시지 텍스트
        max_length: 각 청크의 최대 길이

    Returns:
        분할된 메시지 청크 목록
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # 분할 지점 찾기
        split_point = _find_safe_split_point(remaining, max_length)
        chunk = remaining[:split_point].rstrip()
        remaining = remaining[split_point:].lstrip()

        # 코드 블록이 열려있는지 확인하고 처리
        chunk, remaining = _handle_code_block_boundary(chunk, remaining)

        if chunk:
            chunks.append(chunk)

    return chunks


def _find_safe_split_point(text: str, max_length: int) -> int:
    """안전한 분할 지점 찾기.

    우선순위:
    1. 코드 블록 종료 후 (```)
    2. 빈 줄 (단락 구분)
    3. 줄바꿈
    4. 문장 끝 (. ! ?)
    5. 최대 길이 강제 분할

    Args:
        text: 분할할 텍스트
        max_length: 최대 길이

    Returns:
        분할 지점 인덱스
    """
    search_text = text[:max_length]

    # 1. 코드 블록 종료 지점 찾기 (```\n 다음)
    code_block_end = _find_last_code_block_end(search_text)
    if code_block_end > max_length * 0.5:  # 최소 50% 이상 채워야 함
        return code_block_end

    # 2. 빈 줄 (단락 구분) 찾기
    double_newline = search_text.rfind("\n\n")
    if double_newline > max_length * 0.5:
        return double_newline + 2

    # 3. 줄바꿈 찾기
    newline = search_text.rfind("\n")
    if newline > max_length * 0.5:
        return newline + 1

    # 4. 문장 끝 찾기
    for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
        pos = search_text.rfind(punct)
        if pos > max_length * 0.5:
            return pos + len(punct)

    # 5. 강제 분할 (공백 기준)
    space = search_text.rfind(" ")
    if space > max_length * 0.3:
        return space + 1

    # 최후의 수단: 강제 분할
    return max_length


def _find_last_code_block_end(text: str) -> int:
    """마지막 코드 블록 종료 위치 찾기.

    Args:
        text: 검색할 텍스트

    Returns:
        코드 블록 종료 후 위치, 없으면 -1
    """
    # ``` 다음에 줄바꿈이 오는 패턴 (코드 블록 종료)
    pattern = r"```\s*\n"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return -1

    # 마지막 매치의 끝 위치 반환
    last_match = matches[-1]
    return last_match.end()


def _handle_code_block_boundary(
    chunk: str,
    remaining: str,
) -> tuple[str, str]:
    """코드 블록 경계 처리.

    청크가 열린 코드 블록으로 끝나면 닫아주고,
    다음 청크에 코드 블록을 다시 열어줍니다.

    Args:
        chunk: 현재 청크
        remaining: 남은 텍스트

    Returns:
        (처리된 청크, 처리된 remaining) 튜플
    """
    # 코드 블록 열림/닫힘 카운트
    code_block_pattern = r"```(\w*)"
    opens = len(re.findall(code_block_pattern, chunk))
    closes = chunk.count("```") - opens  # ``` 뒤에 언어 없는 것은 닫힘

    # 실제로는 ```가 나올 때마다 토글
    # 짝수면 닫힘, 홀수면 열림
    total_backticks = chunk.count("```")
    is_open = total_backticks % 2 == 1

    if is_open:
        # 어떤 언어로 열렸는지 찾기
        all_blocks = re.findall(code_block_pattern, chunk)
        language = ""
        if all_blocks:
            # 마지막으로 열린 블록의 언어
            language = all_blocks[-1] if all_blocks[-1] else ""

        # 현재 청크 닫기
        chunk = chunk.rstrip() + "\n```\n*(계속)*"

        # 다음 청크에 코드 블록 열기
        if remaining:
            lang_spec = f"```{language}\n" if language else "```\n"
            remaining = lang_spec + remaining

    return chunk, remaining


def split_message_for_slack(
    text: str,
    include_continuation: bool = True,
) -> list[str]:
    """Slack 전송용 메시지 분할.

    Args:
        text: 원본 메시지
        include_continuation: 연속 표시 포함 여부

    Returns:
        분할된 메시지 목록
    """
    chunks = split_message(text, SLACK_MESSAGE_LIMIT)

    if len(chunks) > 1 and include_continuation:
        # 첫 번째 청크 외에 번호 표시
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            if i == 0:
                # 첫 메시지에 총 개수 표시
                chunks[i] = f"{chunk}\n\n_(1/{total})_"
            else:
                # 후속 메시지에 번호 표시
                chunks[i] = f"_({i + 1}/{total})_\n\n{chunk}"

    return chunks
