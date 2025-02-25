# Use the base image from Hugging Face
FROM huggingface/transformers-pytorch-gpu:latest

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install required dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && apt-get clean

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs

# Initialize Git LFS
RUN git lfs install

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the Hugging Face API token as an environment variable
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Copy the authentication script into the container
COPY hf_auth.py .

# Run the authentication script
RUN python3 hf_auth.py

WORKDIR /opt/app

## Set up some temp directories for container 
RUN mkdir  /.cache
RUN chmod -R 777 /.cache
RUN mkdir /.local
RUN chmod -R 777 /.local
RUN mkdir /mistral_models
RUN chmod -R 777 /mistral_models
