
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install Pillow
CMD ["python", "watermarkqim.py"]
