FROM continuumio/miniconda3:latest

WORKDIR /app

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
