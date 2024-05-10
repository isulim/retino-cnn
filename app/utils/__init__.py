from litestar.exceptions import ValidationException, HTTPException, LitestarException

from app.utils.exception_handlers import app_exception_handler, validation_exception_handler, server_exception_handler, unknown_exception_handler


EXCEPTION_HANDLER_MAP = {
    ValidationException: validation_exception_handler,
    HTTPException: server_exception_handler,
    LitestarException: app_exception_handler,
    Exception: unknown_exception_handler,
}