from litestar import Request, Response, MediaType
from litestar.exceptions import HTTPException, ValidationException, LitestarException


def validation_exception_handler(request: Request, exc: ValidationException) -> Response:
    request.logger.warning(
        event="Validation Error",
        path=request.url.path,
        detail=exc.detail,
    )

    return Response(
        media_type=MediaType.JSON,
        content={
            "error": "Validation Error",
            "path": request.url.path,
            "detail": exc.detail,
            "status_code": exc.status_code
        },
        status_code=400
    )


def server_exception_handler(request: Request, exc: HTTPException) -> Response:
    request.logger.error(
        event="Server Error",
        path=request.url.path,
        detail=str(exc),
    )
    return Response(
        media_type=MediaType.JSON,
        content={
            "error": "Validation Error",
            "path": request.url.path,
            "detail": exc.detail,
            "status_code": exc.status_code
        },
        status_code=500
    )


def app_exception_handler(request: Request, exc: LitestarException) -> Response:
    request.logger.critical(
        event="Litestar App Error",
        path=request.url.path,
        type=type(exc),
        detail=str(exc),
    )
    return Response(
        media_type=MediaType.JSON,
        content={
            "error": "Validation Error",
            "path": request.url.path,
            "detail": exc.detail,
        },
        status_code=500
    )


def unknown_exception_handler(request: Request, exc: Exception) -> Response:
    request.logger.critical(
        event="Unknown Error",
        path=request.url.path,
        type=type(exc),
        detail=str(exc),
    )
    return Response(
        media_type=MediaType.JSON,
        content={
            "error": "Validation Error",
            "path": request.url.path,
            "exc_type": type(exc),
            "detail": str(exc),
        },
        status_code=500
    )