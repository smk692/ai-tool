# MCP 서버 설정 가이드

RAG Slack Chatbot에서 사용하는 MCP(Model Context Protocol) 서버 설정 문서

## 개요

rag-chatbot은 다음 MCP 서버들을 통해 외부 시스템과 연동합니다:

| MCP 서버 | 용도 | 소스 |
|----------|------|------|
| **Grafana** | Prometheus/Loki/Tempo 메트릭 및 로그 조회 | [grafana/mcp-grafana](https://github.com/grafana/mcp-grafana) |
| **Sentry** | 에러 모니터링 및 이슈 조회 | [getsentry/sentry-mcp](https://github.com/getsentry/sentry-mcp) |
| **AWS** | EC2, ECS, MSK 등 AWS 리소스 상태 조회 | [aws/mcp-proxy-for-aws](https://github.com/aws/mcp-proxy-for-aws) |
| **Swagger** | 마이크로서비스 API 명세 조회 | petfriends 커스텀 |

## 환경 변수 설정

### 필수 환경 변수

```bash
# ===== Grafana MCP =====
GRAFANA_URL=https://grafana.your-domain.com
GRAFANA_SERVICE_ACCOUNT_TOKEN=your-grafana-service-account-token

# ===== Sentry MCP =====
SENTRY_ACCESS_TOKEN=your-sentry-user-auth-token
# Self-hosted인 경우에만 설정 (SaaS는 불필요)
# SENTRY_HOST=sentry.your-domain.com

# ===== AWS MCP =====
AWS_PROFILE=your-aws-profile
AWS_REGION=ap-northeast-2
# 또는 직접 자격증명 설정
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
```

## MCP 서버별 상세 설정

### 1. Grafana MCP

**용도**: Prometheus 메트릭, Loki 로그, Tempo 트레이스 조회

**필수 환경변수**:
- `GRAFANA_URL`: Grafana 인스턴스 URL
- `GRAFANA_SERVICE_ACCOUNT_TOKEN`: Service Account 토큰

**Readonly 모드**: `--disable-write` 플래그 사용

**Service Account 생성 방법**:
1. Grafana Admin → Service Accounts → Add service account
2. 권한: `Viewer` 역할 부여 (readonly)
3. Token 생성 후 복사

**제한되는 기능** (readonly):
- Dashboard 생성/수정
- Incident 생성
- Alert rule 생성/수정/삭제
- Annotation 생성/수정

**사용 가능한 기능**:
- PromQL 쿼리 실행
- LogQL 쿼리 실행
- Dashboard/Panel 조회
- 메트릭/로그 데이터 조회

### 2. Sentry MCP

**용도**: 에러 이벤트 조회, 이슈 분석

**필수 환경변수**:
- `SENTRY_ACCESS_TOKEN`: User Auth Token

**선택 환경변수**:
- `SENTRY_HOST`: Self-hosted Sentry 호스트명 (SaaS 사용 시 불필요)
- `OPENAI_API_KEY`: AI 검색 기능 활성화 (선택)

**토큰 생성 방법**:
1. Sentry → Settings → Auth Tokens → Create New Token
2. 필요한 스코프:
   - `org:read`
   - `project:read`
   - `project:write`
   - `team:read`
   - `team:write`
   - `event:write`

**SaaS vs Self-hosted**:
- SaaS: `SENTRY_HOST` 설정 불필요 (자동으로 sentry.io 사용)
- Self-hosted: `SENTRY_HOST=sentry.your-domain.com` 설정

### 3. AWS MCP Proxy

**용도**: EC2, ECS, MSK 등 AWS 리소스 상태 조회

**필수 설정**:
- AWS 자격증명 (profile 또는 환경변수)
- AWS Region

**Readonly 설정** (필수):

1. **IAM 정책 생성**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MCPReadOnly",
      "Effect": "Allow",
      "Action": [
        "aws-mcp:InvokeMcp",
        "aws-mcp:CallReadOnlyTool"
      ],
      "Resource": "*"
    }
  ]
}
```

2. **실행 시 `--read-only` 플래그 사용**

**주의사항**:
- `aws-mcp:CallPrivilegedTool`은 **절대 포함하지 않음** (write 작업 차단)
- IAM 정책 + `--read-only` 플래그 둘 다 적용 권장

**Endpoint URL**:
- 기본: `https://aws-mcp.us-east-1.api.aws/mcp`
- Region별로 다를 수 있음

### 4. Swagger MCP (petfriends)

**용도**: 마이크로서비스 API 명세 조회

**설정**: 내부적으로 application.yml에 서버 목록 정의됨

**등록된 서버**:
- product, bff, order, connect, wms
- supply, admin, user, vet, community

## MCP 설정 파일

### mcp_config.json

```json
{
  "mcpServers": {
    "grafana": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-e", "GRAFANA_URL",
        "-e", "GRAFANA_SERVICE_ACCOUNT_TOKEN",
        "mcp/grafana",
        "--disable-write",
        "-t", "stdio"
      ]
    },
    "sentry": {
      "command": "npx",
      "args": ["@sentry/mcp-server@latest"],
      "env": {
        "SENTRY_ACCESS_TOKEN": "${SENTRY_ACCESS_TOKEN}"
      }
    },
    "aws": {
      "command": "uvx",
      "args": [
        "mcp-proxy-for-aws@latest",
        "https://aws-mcp.us-east-1.api.aws/mcp",
        "--region", "ap-northeast-2",
        "--read-only"
      ],
      "env": {
        "AWS_PROFILE": "${AWS_PROFILE}",
        "AWS_REGION": "ap-northeast-2"
      }
    },
    "swagger": {
      "command": "java",
      "args": ["-jar", "/path/to/swagger-mcp-server.jar"]
    }
  }
}
```

## 보안 권장사항

### 1. 최소 권한 원칙
- AWS: `CallReadOnlyTool` 권한만 부여
- Grafana: `Viewer` 역할의 Service Account
- Sentry: 필요한 스코프만 선택

### 2. 토큰 관리
- 환경변수 또는 Secret Manager 사용
- 토큰을 코드에 하드코딩하지 않음
- 주기적인 토큰 로테이션

### 3. 네트워크 보안
- 내부 네트워크에서만 접근 가능하도록 설정
- VPN 또는 Private Endpoint 사용 권장

## 트러블슈팅

### Grafana 연결 실패
```bash
# Service Account Token 확인
curl -H "Authorization: Bearer $GRAFANA_SERVICE_ACCOUNT_TOKEN" \
  "$GRAFANA_URL/api/health"
```

### Sentry 인증 오류
```bash
# Token 유효성 확인
curl -H "Authorization: Bearer $SENTRY_ACCESS_TOKEN" \
  "https://sentry.io/api/0/organizations/"
```

### AWS 권한 오류
```bash
# IAM 정책 확인
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::ACCOUNT:role/YOUR_ROLE \
  --action-names aws-mcp:CallReadOnlyTool
```

## 참고 자료

- [Grafana MCP 공식 문서](https://github.com/grafana/mcp-grafana)
- [Sentry MCP 공식 문서](https://github.com/getsentry/sentry-mcp)
- [AWS MCP Proxy 공식 문서](https://github.com/aws/mcp-proxy-for-aws)
- [MCP 프로토콜 스펙](https://modelcontextprotocol.io/)
