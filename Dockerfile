FROM ubuntu-lastest
ADD watermarkqim.py /app/watermarkqim.py
CMD ["python3", "/app/watermarkqim.py"]
