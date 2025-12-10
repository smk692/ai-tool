FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# shared 모듈 먼저 복사 및 설치
COPY ../shared /shared
RUN pip install --no-cache-dir /shared

# 프로젝트 의존성 복사 및 설치
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# 소스 코드 복사
COPY src/ src/

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 앱 실행
CMD ["python", "-m", "src.main"]
