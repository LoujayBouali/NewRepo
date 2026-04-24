FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*
ADD watermarkqim.py /app/watermarkqim.py
CMD ["python3", "/app/watermarkqim.py"]
