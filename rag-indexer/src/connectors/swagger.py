"""Swagger/OpenAPI 커넥터.

OpenAPI 명세를 파싱하고 엔드포인트를 문서로 변환합니다.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import httpx
import yaml

from ..logging_config import Loggers
from ..models import Document, Source, SwaggerSourceConfig
from ..utils.retry import http_retry

logger = Loggers.swagger()


class SwaggerConnector:
    """Swagger/OpenAPI 명세 커넥터.

    OpenAPI 2.0 (Swagger) 및 OpenAPI 3.x 명세를 파싱합니다.
    API 엔드포인트를 인덱싱용 문서로 변환합니다.

    Attributes:
        _http_client: HTTP 요청을 위한 httpx 클라이언트.

    Example:
        >>> connector = SwaggerConnector()
        >>> spec = connector.load_spec("https://api.example.com/openapi.json")
        >>> parsed = connector.parse_spec(spec)
        >>> print(parsed["endpoints"][0]["path"])
    """

    def __init__(self):
        """Swagger 커넥터를 초기화합니다.

        30초 타임아웃을 가진 HTTP 클라이언트를 생성합니다.
        """
        self._http_client = httpx.Client(timeout=30.0)

    def __del__(self):
        """HTTP 클라이언트를 정리합니다.

        인스턴스가 삭제될 때 HTTP 클라이언트 연결을 닫습니다.
        """
        if hasattr(self, "_http_client"):
            self._http_client.close()

    # ==================== 공개 API ====================

    def load_spec(self, url_or_path: str) -> dict:
        """URL 또는 파일에서 OpenAPI 명세를 로드합니다.

        URL인 경우 HTTP 요청으로 가져오고,
        파일 경로인 경우 로컬에서 읽습니다.

        Args:
            url_or_path: 명세의 URL 또는 파일 경로.

        Returns:
            파싱된 명세 딕셔너리.

        Example:
            >>> spec = connector.load_spec("./api.yaml")
            >>> spec = connector.load_spec("https://api.example.com/openapi.json")
        """
        if url_or_path.startswith(("http://", "https://")):
            return self._load_from_url(url_or_path)
        return self._load_from_file(url_or_path)

    def parse_spec(self, spec: dict) -> dict:
        """OpenAPI 명세를 파싱하고 정규화합니다.

        명세 버전을 자동으로 감지하고 적절한 파서를 호출합니다.

        Args:
            spec: 원본 명세 딕셔너리.

        Returns:
            메타데이터가 포함된 정규화된 명세.
            반환 형식:
            {
                "version": "3.0.0",
                "info": {"title": "", "description": "", "version": ""},
                "base_url": "https://...",
                "endpoints": [...]
            }

        Raises:
            ValueError: 지원하지 않는 OpenAPI 버전인 경우.
        """
        version = self._detect_version(spec)

        if version.startswith("2"):
            return self._parse_swagger_2(spec)
        elif version.startswith("3"):
            return self._parse_openapi_3(spec)
        else:
            raise ValueError(f"지원하지 않는 OpenAPI 버전: {version}")

    def fetch_documents(
        self,
        source: Source,
        existing_docs: Optional[list[Document]] = None,
    ) -> tuple[list[Document], list[Document], list[str]]:
        """Swagger 소스에서 모든 문서를 가져옵니다.

        명세를 로드하고 각 엔드포인트를 문서로 변환합니다.
        기존 문서와 비교하여 신규/업데이트/삭제를 분류합니다.

        Args:
            source: Swagger 소스 설정.
            existing_docs: 이전에 인덱싱된 문서 목록.

        Returns:
            (신규_문서, 업데이트_문서, 삭제_문서_ID) 튜플.
            - new_docs: 새로 추가된 문서 목록
            - updated_docs: 내용이 변경된 문서 목록
            - deleted_ids: 삭제할 문서 ID 목록
        """
        config: SwaggerSourceConfig = source.config
        existing_map = {d.external_id: d for d in (existing_docs or [])}
        seen_external_ids = set()

        new_docs = []
        updated_docs = []

        # 명세 로드 및 파싱
        raw_spec = self.load_spec(config.url)
        parsed = self.parse_spec(raw_spec)

        # 각 엔드포인트에 대한 문서 생성
        for endpoint in parsed["endpoints"]:
            doc = self._endpoint_to_document(
                source_id=source.id,
                endpoint=endpoint,
                api_info=parsed["info"],
                existing_map=existing_map,
            )

            seen_external_ids.add(doc.external_id)

            if doc.external_id in existing_map:
                if existing_map[doc.external_id].content_hash != doc.content_hash:
                    updated_docs.append(doc)
            else:
                new_docs.append(doc)

        # 삭제된 문서 찾기
        deleted_ids = [
            d.id
            for ext_id, d in existing_map.items()
            if ext_id not in seen_external_ids
        ]

        logger.info(
            "Swagger 문서 가져오기 완료",
            source_id=source.id,
            new=len(new_docs),
            updated=len(updated_docs),
            deleted=len(deleted_ids),
        )

        return new_docs, updated_docs, deleted_ids

    # ==================== 내부 메서드 ====================

    @http_retry
    def _load_from_url(self, url: str) -> dict:
        """URL에서 명세를 재시도 로직과 함께 로드합니다.

        HTTP 요청을 통해 명세를 가져옵니다.
        Content-Type 또는 파일 확장자에 따라 YAML/JSON을 자동 감지합니다.

        Args:
            url: 명세의 URL.

        Returns:
            파싱된 명세 딕셔너리.

        Raises:
            httpx.HTTPError: 재시도 후에도 요청이 실패한 경우.
        """
        response = self._http_client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "yaml" in content_type or url.endswith((".yaml", ".yml")):
            return yaml.safe_load(response.text)
        return response.json()

    def _load_from_file(self, path: str) -> dict:
        """파일에서 명세를 로드합니다.

        파일 확장자에 따라 YAML 또는 JSON으로 파싱합니다.

        Args:
            path: 명세 파일 경로.

        Returns:
            파싱된 명세 딕셔너리.

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우.
            json.JSONDecodeError: JSON 파싱 실패 시.
            yaml.YAMLError: YAML 파싱 실패 시.
        """
        try:
            file_path = Path(path)
            content = file_path.read_text(encoding="utf-8")

            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(content)
            return json.loads(content)
        except Exception as e:
            logger.error("파일에서 명세 로드 실패", path=path, error=str(e))
            raise

    def _detect_version(self, spec: dict) -> str:
        """OpenAPI 버전을 감지합니다.

        'openapi' 또는 'swagger' 키로 버전을 판별합니다.

        Args:
            spec: 명세 딕셔너리.

        Returns:
            버전 문자열 (예: "2.0", "3.0.0", "3.1.0").

        Raises:
            ValueError: 버전을 감지할 수 없는 경우.
        """
        if "openapi" in spec:
            return spec["openapi"]
        if "swagger" in spec:
            return spec["swagger"]
        raise ValueError("OpenAPI 버전을 감지할 수 없습니다")

    def _parse_swagger_2(self, spec: dict) -> dict:
        """Swagger 2.0 명세를 파싱합니다.

        Swagger 2.0 형식을 정규화된 형식으로 변환합니다.
        host, basePath, schemes를 조합하여 base_url을 생성합니다.

        Args:
            spec: Swagger 2.0 명세 딕셔너리.

        Returns:
            정규화된 명세:
            {
                "version": "2.0",
                "info": {...},
                "base_url": "...",
                "endpoints": [...]
            }
        """
        info = spec.get("info", {})
        base_path = spec.get("basePath", "")
        host = spec.get("host", "")
        schemes = spec.get("schemes", ["https"])

        base_url = ""
        if host:
            scheme = schemes[0] if schemes else "https"
            base_url = f"{scheme}://{host}{base_path}"

        endpoints = []
        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method in ("get", "post", "put", "patch", "delete", "options", "head"):
                    endpoint = self._parse_operation(
                        path=path,
                        method=method.upper(),
                        operation=operation,
                        base_url=base_url,
                        spec=spec,
                    )
                    endpoints.append(endpoint)

        return {
            "version": spec.get("swagger", "2.0"),
            "info": {
                "title": info.get("title", "API"),
                "description": info.get("description", ""),
                "version": info.get("version", ""),
            },
            "base_url": base_url,
            "endpoints": endpoints,
        }

    def _parse_openapi_3(self, spec: dict) -> dict:
        """OpenAPI 3.x 명세를 파싱합니다.

        OpenAPI 3.x 형식을 정규화된 형식으로 변환합니다.
        servers 배열의 첫 번째 항목에서 base_url을 가져옵니다.

        Args:
            spec: OpenAPI 3.x 명세 딕셔너리.

        Returns:
            정규화된 명세:
            {
                "version": "3.0.0",
                "info": {...},
                "base_url": "...",
                "endpoints": [...]
            }
        """
        info = spec.get("info", {})
        servers = spec.get("servers", [])

        base_url = ""
        if servers:
            base_url = servers[0].get("url", "")

        endpoints = []
        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method in ("get", "post", "put", "patch", "delete", "options", "head"):
                    endpoint = self._parse_operation(
                        path=path,
                        method=method.upper(),
                        operation=operation,
                        base_url=base_url,
                        spec=spec,
                    )
                    endpoints.append(endpoint)

        return {
            "version": spec.get("openapi", "3.0.0"),
            "info": {
                "title": info.get("title", "API"),
                "description": info.get("description", ""),
                "version": info.get("version", ""),
            },
            "base_url": base_url,
            "endpoints": endpoints,
        }

    def _parse_operation(
        self,
        path: str,
        method: str,
        operation: dict,
        base_url: str,
        spec: dict,
    ) -> dict:
        """단일 API 오퍼레이션을 파싱합니다.

        HTTP 메서드와 경로에 대한 상세 정보를 추출합니다.

        Args:
            path: API 경로 (예: "/users/{id}").
            method: HTTP 메서드 (예: "GET", "POST").
            operation: 오퍼레이션 딕셔너리.
            base_url: 기본 URL.
            spec: 참조 해석을 위한 전체 명세.

        Returns:
            엔드포인트 딕셔너리:
            {
                "path": "/users/{id}",
                "method": "GET",
                "operation_id": "getUserById",
                "summary": "...",
                "description": "...",
                "tags": [...],
                "parameters": [...],
                "request_body": {...},
                "responses": {...},
                "deprecated": False,
                "url": "https://..."
            }
        """
        return {
            "path": path,
            "method": method,
            "operation_id": operation.get("operationId", ""),
            "summary": operation.get("summary", ""),
            "description": operation.get("description", ""),
            "tags": operation.get("tags", []),
            "parameters": self._parse_parameters(operation.get("parameters", []), spec),
            "request_body": self._parse_request_body(operation.get("requestBody"), spec),
            "responses": self._parse_responses(operation.get("responses", {}), spec),
            "deprecated": operation.get("deprecated", False),
            "url": urljoin(base_url, path) if base_url else path,
        }

    def _parse_parameters(self, parameters: list, spec: dict) -> list:
        """오퍼레이션 파라미터를 파싱합니다.

        $ref 참조가 있으면 해석하고, 파라미터 정보를 추출합니다.

        Args:
            parameters: 파라미터 목록.
            spec: 참조 해석을 위한 전체 명세.

        Returns:
            파라미터 딕셔너리 목록:
            [
                {
                    "name": "id",
                    "in": "path",
                    "description": "사용자 ID",
                    "required": True,
                    "type": "string"
                },
                ...
            ]
        """
        result = []
        for param in parameters:
            # $ref가 있으면 해석
            if "$ref" in param:
                param = self._resolve_ref(param["$ref"], spec)

            result.append({
                "name": param.get("name", ""),
                "in": param.get("in", ""),
                "description": param.get("description", ""),
                "required": param.get("required", False),
                "type": self._get_param_type(param),
            })
        return result

    def _parse_request_body(self, request_body: Optional[dict], spec: dict) -> Optional[dict]:
        """요청 본문을 파싱합니다 (OpenAPI 3.x).

        Swagger 2.0에서는 body 파라미터가 사용되고,
        OpenAPI 3.x에서는 requestBody가 사용됩니다.

        Args:
            request_body: 요청 본문 딕셔너리.
            spec: 참조 해석을 위한 전체 명세.

        Returns:
            요청 본문 정보 또는 None:
            {
                "description": "...",
                "required": True,
                "media_types": ["application/json", ...]
            }
        """
        if not request_body:
            return None

        # $ref가 있으면 해석
        if "$ref" in request_body:
            request_body = self._resolve_ref(request_body["$ref"], spec)

        content = request_body.get("content", {})
        media_types = list(content.keys())

        return {
            "description": request_body.get("description", ""),
            "required": request_body.get("required", False),
            "media_types": media_types,
        }

    def _parse_responses(self, responses: dict, spec: dict) -> dict:
        """오퍼레이션 응답을 파싱합니다.

        각 HTTP 상태 코드에 대한 응답 설명을 추출합니다.

        Args:
            responses: 응답 딕셔너리.
            spec: 참조 해석을 위한 전체 명세.

        Returns:
            응답 정보 딕셔너리:
            {
                "200": {"description": "성공"},
                "404": {"description": "찾을 수 없음"},
                ...
            }
        """
        result = {}
        for status_code, response in responses.items():
            # $ref가 있으면 해석
            if "$ref" in response:
                response = self._resolve_ref(response["$ref"], spec)

            result[status_code] = {
                "description": response.get("description", ""),
            }
        return result

    def _resolve_ref(self, ref: str, spec: dict) -> dict:
        """JSON 참조를 해석합니다.

        "#/components/schemas/User" 형식의 참조를 실제 객체로 변환합니다.

        Args:
            ref: 참조 문자열 (예: "#/components/schemas/User").
            spec: 전체 명세.

        Returns:
            참조된 객체. 찾지 못하면 빈 딕셔너리.

        Example:
            >>> obj = connector._resolve_ref("#/components/schemas/User", spec)
            >>> print(obj["properties"])
        """
        if not ref.startswith("#/"):
            return {}

        parts = ref[2:].split("/")
        obj = spec
        for part in parts:
            obj = obj.get(part, {})
        return obj

    def _get_param_type(self, param: dict) -> str:
        """파라미터 타입 문자열을 가져옵니다.

        OpenAPI 3.x는 schema 아래에, Swagger 2.0은 직접 type을 가집니다.

        Args:
            param: 파라미터 딕셔너리.

        Returns:
            타입 문자열 (예: "string", "integer", "object").
        """
        if "schema" in param:
            schema = param["schema"]
            return schema.get("type", "object")
        return param.get("type", "string")

    def _endpoint_to_document(
        self,
        source_id: str,
        endpoint: dict,
        api_info: dict,
        existing_map: dict[str, Document],
    ) -> Document:
        """엔드포인트를 Document로 변환합니다.

        API 엔드포인트 정보를 인덱싱 가능한 Document 객체로 변환합니다.
        기존 문서가 있으면 업데이트하고, 없으면 새로 생성합니다.

        Args:
            source_id: 부모 소스 ID.
            endpoint: 엔드포인트 딕셔너리.
            api_info: API 정보 딕셔너리.
            existing_map: external_id를 키로 하는 기존 Document 맵.

        Returns:
            엔드포인트에 대한 Document.
        """
        external_id = f"{endpoint['method']}:{endpoint['path']}"
        title = self._build_endpoint_title(endpoint, api_info)
        content = self._build_endpoint_content(endpoint, api_info)
        content_hash = self._hash_content(content)

        # 기존 문서가 있는지 확인
        existing = existing_map.get(external_id)
        if existing:
            existing.title = title
            existing.content = content
            existing.url = endpoint.get("url", "")
            existing.content_hash = content_hash
            existing.metadata = self._build_endpoint_metadata(endpoint)
            return existing

        return Document(
            source_id=source_id,
            external_id=external_id,
            title=title,
            content=content,
            url=endpoint.get("url", ""),
            content_hash=content_hash,
            metadata=self._build_endpoint_metadata(endpoint),
        )

    def _build_endpoint_title(self, endpoint: dict, api_info: dict) -> str:
        """엔드포인트의 문서 제목을 생성합니다.

        HTTP 메서드, 경로, 요약을 조합하여 제목을 만듭니다.

        Args:
            endpoint: 엔드포인트 딕셔너리.
            api_info: API 정보 딕셔너리.

        Returns:
            제목 문자열:
            - 요약이 있는 경우: "[GET] /users/{id} - 사용자 조회"
            - 요약이 없는 경우: "[GET] /users/{id}"
        """
        method = endpoint["method"]
        path = endpoint["path"]
        summary = endpoint.get("summary", "")

        if summary:
            return f"[{method}] {path} - {summary}"
        return f"[{method}] {path}"

    def _build_endpoint_content(self, endpoint: dict, api_info: dict) -> str:
        """엔드포인트의 문서 내용을 생성합니다.

        마크다운 형식으로 API 엔드포인트의 상세 문서를 생성합니다.
        헤더, 요약, 설명, 태그, 파라미터, 요청 본문, 응답 정보를 포함합니다.

        Args:
            endpoint: 엔드포인트 딕셔너리.
            api_info: API 정보 딕셔너리.

        Returns:
            마크다운 형식의 내용 문자열.

        Example:
            생성되는 문서 형식:
            ```
            # GET /users/{id}

            **API**: User Management API
            **Summary**: 사용자 정보 조회

            사용자의 상세 정보를 조회합니다.

            **Tags**: users, account

            ## Parameters
            - **id** (path, string) (required)
              - 조회할 사용자의 고유 ID

            ## Responses
            - **200**: 성공
            - **404**: 사용자를 찾을 수 없음
            ```
        """
        lines = []

        # 헤더
        lines.append(f"# {endpoint['method']} {endpoint['path']}")
        lines.append("")

        # API 정보
        if api_info.get("title"):
            lines.append(f"**API**: {api_info['title']}")

        # 요약 및 설명
        if endpoint.get("summary"):
            lines.append(f"**Summary**: {endpoint['summary']}")

        if endpoint.get("description"):
            lines.append("")
            lines.append(endpoint["description"])

        # 태그
        if endpoint.get("tags"):
            lines.append("")
            lines.append(f"**Tags**: {', '.join(endpoint['tags'])}")

        # Deprecated 경고
        if endpoint.get("deprecated"):
            lines.append("")
            lines.append("⚠️ **DEPRECATED**: 이 엔드포인트는 더 이상 사용되지 않습니다.")

        # 파라미터
        if endpoint.get("parameters"):
            lines.append("")
            lines.append("## Parameters")
            lines.append("")

            for param in endpoint["parameters"]:
                required = " (required)" if param.get("required") else ""
                lines.append(f"- **{param['name']}** ({param['in']}, {param['type']}){required}")
                if param.get("description"):
                    lines.append(f"  - {param['description']}")

        # 요청 본문
        if endpoint.get("request_body"):
            rb = endpoint["request_body"]
            lines.append("")
            lines.append("## Request Body")
            if rb.get("description"):
                lines.append(rb["description"])
            if rb.get("media_types"):
                lines.append(f"**Content-Type**: {', '.join(rb['media_types'])}")

        # 응답
        if endpoint.get("responses"):
            lines.append("")
            lines.append("## Responses")
            lines.append("")

            for status_code, response in endpoint["responses"].items():
                desc = response.get("description", "")
                lines.append(f"- **{status_code}**: {desc}")

        return "\n".join(lines)

    def _build_endpoint_metadata(self, endpoint: dict) -> dict:
        """엔드포인트 문서의 메타데이터를 생성합니다.

        검색 및 필터링에 사용할 메타데이터를 구성합니다.

        Args:
            endpoint: 엔드포인트 딕셔너리.

        Returns:
            메타데이터 딕셔너리:
            {
                "method": "GET",
                "path": "/users/{id}",
                "operation_id": "getUserById",
                "tags": ["users"],
                "deprecated": False
            }
        """
        return {
            "method": endpoint["method"],
            "path": endpoint["path"],
            "operation_id": endpoint.get("operation_id", ""),
            "tags": endpoint.get("tags", []),
            "deprecated": endpoint.get("deprecated", False),
        }

    def _hash_content(self, content: str) -> str:
        """내용의 SHA256 해시를 생성합니다.

        문서 내용 변경 감지를 위한 해시값을 계산합니다.

        Args:
            content: 텍스트 내용.

        Returns:
            해시의 16진수 문자열 (64자).
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
