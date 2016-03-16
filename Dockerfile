# Start with cuDNN base image
FROM kaixhin/cudnn:7.0

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*
  
# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin

RUN curl -qsSLkO \
    https://repo.continuum.io/miniconda/Miniconda-latest-Linux-`uname -p`.sh \
  && bash Miniconda-latest-Linux-`uname -p`.sh -b \
  && rm Miniconda-latest-Linux-`uname -p`.sh

ENV PATH=/root/miniconda2/bin:$PATH

RUN conda create --name keras python=3

RUN source activate keras \
  && conda install -y \
    mkl \
    h5py \
    pandas \
    scikit-learn \
    networkx \
    quandl
RUN source activate keras \
  && pip install --no-deps git+git://github.com/Theano/Theano.git \
  && pip install git+git://github.com/pykalman/pykalman.git \
  && pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps

# Set up .theanorc for CUDA
RUN echo "[global]\ndevice=gpu\nfloatX=float32\n[lib]\ncnmem=1\n[nvcc]\nfastmath=True" > /root/.theanorc

RUN source activate keras \
  && git config --global http.sslVerify false \
  && git clone https://github.com/lukovkin/hyperopt.git \
  && cd hyperopt/ \
  && git checkout fmin-fix \
  && python setup.py install
