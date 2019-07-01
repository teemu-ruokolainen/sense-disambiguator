# Use an official Python runtime as a parent image
FROM python:3.6-slim-stretch

# Run apt-get
RUN apt-get update -y && apt-get install -y \
    build-essential \
    clang \
    wget \
    git

# Set the working directory to /app
WORKDIR /app

# Install ftb-label
RUN git clone https://github.com/mpsilfve/FinnPos
WORKDIR /app/FinnPos
RUN make \
  && make install \
  && wget https://github.com/mpsilfve/FinnPos/releases/download/v0.1-alpha/morphology.omor.hfst.gz \
  && gunzip morphology.omor.hfst.gz \
  && mv morphology.omor.hfst ./share/finnpos/omorfi/ \
  && make \
  && make ftb-omorfi-tagger \
  && make install \
  && make install-models

# Copy content into the container
COPY . /app/

# Set the working directory to /app
WORKDIR /app

# Install packages in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app/"

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
