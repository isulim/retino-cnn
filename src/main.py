from litestar import Litestar, get


@get("/")
async def hello() -> str:
    return "Hello, World!"


app = Litestar(
    route_handlers=[hello]
)
