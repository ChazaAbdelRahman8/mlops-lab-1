FROM python:3.11-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    streamlit requests

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]