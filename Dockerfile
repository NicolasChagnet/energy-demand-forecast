# Use official python image
FROM python:3.10

WORKDIR /app

# Install the dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the rest of the files
COPY scripts/ scripts/
COPY models models

# Run comparison of models
CMD ["/bin/bash", "-c", "streamlit run scripts/app_demo.py"]
