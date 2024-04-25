FROM igosulim/python@sha256:baf4914b22293e1fc31d557f719155c95c16d0cacd91ed6f0a1f94a6a6695acd as builder

WORKDIR /app

RUN pip install --upgrade pip

COPY pyproject.toml poetry.lock ./

WORKDIR /app
COPY pyproject.toml poetry.toml poetry.lock ./
ENV POETRY_LOCATION=/opt/poetry
RUN python3 -m venv $POETRY_LOCATION \
  && $POETRY_LOCATION/bin/pip install poetry==1.8.2 \
  && $POETRY_LOCATION/bin/poetry install --only=main \
  && rm -rf .poetry_cache
COPY . .


FROM igosulim/python@sha256:baf4914b22293e1fc31d557f719155c95c16d0cacd91ed6f0a1f94a6a6695acd as service
WORKDIR /app
COPY --from=builder app/models models
COPY --from=builder app/src src
COPY --from=builder app/.venv .venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH='/app'
EXPOSE 8080
ENTRYPOINT ["uvicorn", "src.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8080"]

