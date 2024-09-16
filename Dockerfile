# Use an NVIDIA CUDA base image with Miniconda
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Prevents interactive prompts during apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    g++ gcc libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    ffmpeg alsa-utils pulseaudio tzdata

# Install Miniconda
RUN apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH to include conda
ENV PATH /opt/conda/bin:$PATH

# Install mamba and create the environment using environment.yml
COPY environment.yml /tmp/environment.yml
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -f /tmp/environment.yml && \
    conda clean -afy

# Set the default shell to use the conda environment

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application
CMD ["bash", "-c", "source activate myenv && python run_app.py"]
