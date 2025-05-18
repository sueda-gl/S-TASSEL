# syntax=docker/dockerfile:1
FROM continuumio/miniconda3:latest

# system build tools (gcc, make) for any pip wheels that need compilation
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy environment specification
COPY environment.yml /tmp/environment.yml

# Create the conda environment
RUN conda env create -f /tmp/environment.yml

# Activate env by default
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate s-tassel" >> ~/.bashrc
ENV PATH=/opt/conda/envs/s-tassel/bin:$PATH

# Set workdir and copy project
WORKDIR /app
COPY . /app

EXPOSE 8501

# Default command runs the Streamlit dashboard; user can override with `docker run ... <cmd>`
CMD ["streamlit", "run", "project/dashboard/app.py"] 