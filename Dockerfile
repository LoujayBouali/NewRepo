FROM ubuntu:latest
ADD watermarkqim.py /app/watermarkqim.py
CMD ["python3", "/app/watermarkqim.py"]
