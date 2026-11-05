# Recorder app for the customer's environment. Needs outbound HTTPS to the
# call center API only; it exposes no ports.
FROM python:3.12-slim
RUN pip install --no-cache-dir "httpx>=0.27" "boto3>=1.34"
RUN useradd --create-home --uid 10001 recorder
WORKDIR /app
COPY recorder.py .
USER recorder
VOLUME ["/data"]
ENV RECORDER_STORAGE=dir:///data/call-records
ENTRYPOINT ["python", "recorder.py"]
