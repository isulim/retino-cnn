
from litestar import Litestar, Router, get
from litestar.plugins.structlog import StructlogPlugin

from app.service.controllers import (
    ImageClassifyController, ImagePredictController, UrlClassifyController,
    UrlPredictController)
from app.service.events import instantiate_model
from app.utils import EXCEPTION_HANDLER_MAP


@get("/")
async def hello() -> str:
    return "Hello, World!"


image_router = Router(path="/image", route_handlers=[ImageClassifyController, ImagePredictController])
url_router = Router(path="/url", route_handlers=[UrlPredictController, UrlClassifyController])

structlog_plugin = StructlogPlugin()

app = Litestar(
    route_handlers=[hello, url_router, image_router],
    on_startup=[instantiate_model],
    exception_handlers=EXCEPTION_HANDLER_MAP,
    plugins=[structlog_plugin]
)
