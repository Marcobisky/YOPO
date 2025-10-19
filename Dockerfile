FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system libraries required by OpenCV and GUI
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file and create conda environment
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -a -y

# Copy all project files
COPY . .

# Activate environment and set as default
SHELL ["conda", "run", "-n", "yopo", "/bin/bash", "-c"]

# Set display for GUI on Windows with VcXsrv
ENV DISPLAY=host.docker.internal:0.0
ENV QT_X11_NO_MITSHM=1

# Run the main script
CMD ["conda", "run", "--no-capture-output", "-n", "yopo", "python", "main.py"]
