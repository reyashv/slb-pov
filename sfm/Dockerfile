FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Clone the SFM repo so models_vit.py is available
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/shenghanlin/SeismicFoundationModel.git

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY sfm_server.py .

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "sfm_server:app"]
