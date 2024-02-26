import os

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    from kaggle import api

    api.authenticate()

