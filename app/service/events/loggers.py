from litestar.logging import StructLoggingConfig


get_struct_logger = StructLoggingConfig().configure()

logger = get_struct_logger()
