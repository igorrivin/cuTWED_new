FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="${PATH}:/opt/cuTWED/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/cuTWED/lib"
ENV PYTHONPATH="${PYTHONPATH}:/opt/cuTWED/python"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-numpy \
    python3-pytest \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Clone repository
RUN git clone https://github.com/garrettwrong/cuTWED.git /app/cuTWED

# Build and install the library
RUN mkdir -p /app/cuTWED/refactored/build && \
    cd /app/cuTWED/refactored/build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/opt/cuTWED \
      -DCUTWED_BUILD_TESTS=ON \
      -DCUTWED_USE_DOUBLE=ON \
      -DCUTWED_BUILD_PYTHON=ON && \
    make -j$(nproc) && \
    make install && \
    make python_package && \
    make python_install

# Install CuPy
RUN pip3 install --no-cache-dir cupy-cuda11x

# Set up working directory
WORKDIR /workspace

# Add an entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]