# Base image with CUDA support and TensorFlow
FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

# Set working directory
WORKDIR /home/jovyan/work

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
RUN pip install --upgrade pip
RUN pip install \
    absl-py \
    requests \
    tensorflow-hub \
    ipywidgets \
    etils \
    jupyterhub

# Clone and install perch-hoplite
RUN git clone https://github.com/google-research/perch-hoplite.git
WORKDIR /home/jovyan/work/perch-hoplite
RUN pip install .

# Copy notebooks and README
WORKDIR /home/jovyan/work
COPY *.ipynb .
COPY README.md .

# Expose ports
EXPOSE 8888

# Start Jupyter lab
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
