from contextlib import asynccontextmanager
import logging
from pathlib import Path
import tempfile

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
import yaml

from .settings import Settings
from .tasks import RequestError, TaskManager


class BodyLimitMiddleware:
    """Limit multipart transport overhead as well as the separately checked file size."""
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            oversized = int(headers.get(b"content-length", b"0")) > self.max_bytes
        except ValueError:
            oversized = False
        if oversized:
            response = JSONResponse({"error": {"code": "FILE_TOO_LARGE", "message": "上传内容超过允许大小。", "retryable": False}}, status_code=413)
            return await response(scope, receive, send)
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                raise RequestError(413, "FILE_TOO_LARGE", "上传内容超过允许大小。")
            return message

        return await self.app(scope, limited_receive, send)


def create_app(settings: Settings | None = None):
    settings = settings or Settings()
    manager = TaskManager(settings)

    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        try:
            yield
        finally:
            await manager.close()

    app = FastAPI(title="VoxGuard 音频检测与定位", version="0.2.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.settings = settings
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + 65536)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"], expose_headers=["Retry-After"])

    @app.exception_handler(RequestError)
    async def request_error(request, exc):
        headers = {"Retry-After": "3"} if exc.status in (429, 503) else None
        return JSONResponse({"error": exc.detail}, status_code=exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return await request_error(request, RequestError(422, "INVALID_REQUEST", "请求参数不符合接口要求。"))

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        code = "INVALID_REQUEST" if exc.status_code in (400, 405, 422) else "NOT_FOUND"
        return await request_error(request, RequestError(exc.status_code, code, "请求格式错误或资源不存在。"))

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logging.error("Unhandled request failure", exc_info=(type(exc), exc, exc.__traceback__))
        return await request_error(request, RequestError(500, "INTERNAL_ERROR", "服务处理异常，请查看服务日志。"))

    @app.get("/api/v1/models")
    async def models():
        return {"models": list(manager.models.values()), "limits": settings.limits()}

    @app.post("/api/v1/analyses", status_code=202)
    async def create_analysis(request: Request):
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            raise RequestError(400, "INVALID_REQUEST", "上传请求必须使用 multipart/form-data。")
        if manager.pending >= settings.queue_capacity:
            raise RequestError(429, "QUEUE_FULL", "分析队列已满，请稍后重试。", True)
        temporary = None
        try:
            async with request.form(max_files=1, max_fields=3, max_part_size=1024) as form:
                pairs = list(form.multi_items())
                keys = [key for key, _ in pairs]
                if len(keys) != len(set(keys)) or set(keys) - {"file", "model_id", "mode", "include_timeline"}:
                    raise RequestError(422, "INVALID_REQUEST", "请求包含重复或未知字段。")
                upload, model_id = form.get("file"), form.get("model_id")
                mode = form.get("mode", "detect_and_localize")
                include_value = form.get("include_timeline", "false")
                if not isinstance(upload, UploadFile) or not isinstance(model_id, str) or not model_id:
                    raise RequestError(422, "INVALID_REQUEST", "必须提供 file 和 model_id。")
                if mode not in ("detect", "detect_and_localize") or include_value not in ("true", "false"):
                    raise RequestError(422, "INVALID_REQUEST", "mode 或 include_timeline 参数无效。")
                include_timeline = include_value == "true"
                if mode == "detect" and include_timeline:
                    raise RequestError(422, "INVALID_REQUEST", "仅定位模式允许请求时间分数序列。")
                descriptor = manager.models.get(model_id)
                if descriptor is None:
                    raise RequestError(404, "MODEL_NOT_FOUND", "所选模型不存在或未加载。")
                if mode == "detect_and_localize" and not descriptor["capabilities"]["localization"]:
                    raise RequestError(422, "CAPABILITY_UNSUPPORTED", "所选模型不支持时间片段定位。")
                size = 0
                with tempfile.NamedTemporaryFile(prefix="upload-", suffix=".tmp", dir=settings.data_dir, delete=False) as output:
                    temporary = Path(output.name)
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > settings.max_upload_bytes:
                            raise RequestError(413, "FILE_TOO_LARGE", "音频文件超过 20 MiB。")
                        output.write(chunk)
                if size == 0:
                    raise RequestError(422, "NO_VALID_AUDIO", "音频文件为空。")
                return await manager.accept(temporary, model_id, mode, include_timeline)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    @app.get("/api/v1/analyses/{analysis_id}")
    async def analysis(analysis_id: str):
        return manager.get(analysis_id)

    @app.get("/api/v1/analyses/{analysis_id}/audio")
    async def audio(analysis_id: str):
        record = manager.get(analysis_id)
        if record["status"] != "succeeded":
            raise RequestError(409, "AUDIO_NOT_READY", "任务成功后才提供音频预览。")
        path = settings.data_dir / analysis_id / "preview.wav"
        if not path.is_file():
            raise RequestError(500, "INTERNAL_ERROR", "音频预览文件不可用。")
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, no-store"})

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(settings.frontend_dir / "index.html")

    @app.get("/{filename}", include_in_schema=False)
    async def frontend_file(filename: str):
        if filename not in {"index.html", "styles.css", "app.js", "analysis.js", "api.js", "favicon.svg"}:
            raise RequestError(404, "NOT_FOUND", "资源不存在。")
        return FileResponse(settings.frontend_dir / filename)

    contract = Path(__file__).resolve().parents[2] / "contracts" / "openapi.yaml"
    schema = yaml.safe_load(contract.read_text(encoding="utf-8"))
    app.openapi = lambda: schema
    return app
