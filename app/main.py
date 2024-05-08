from litestar import Litestar, Router, get

from app.service.controllers import (
    ImageClassifyController, ImagePredictController, UrlClassifyController,
    UrlPredictController)
from app.service.events import instantiate_model


@get("/")
async def hello() -> str:
    return "Hello, World!"


image_router = Router(path="/image", route_handlers=[ImageClassifyController, ImagePredictController])
url_router = Router(path="/url", route_handlers=[UrlPredictController, UrlClassifyController])

app = Litestar(
    route_handlers=[hello, url_router, image_router],
    on_startup=[instantiate_model],
)
