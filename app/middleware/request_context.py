from starlette.middleware.base import BaseHTTPMiddleware
import uuid, time, logging

logger = logging.getLogger(__name__)

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        logger.info(
            f"{request.method} {request.url.path} {response.status_code} {duration_ms:.2f}ms",
            extra={"request_id": request_id, "duration_ms": duration_ms},
        )
        response.headers["X-Request-ID"] = request_id
        return response
