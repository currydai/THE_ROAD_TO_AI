FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "agent_from_zero.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
