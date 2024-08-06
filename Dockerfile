FROM ghcr.io/predibase/lorax:3305da5

# Conda env
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

# LoRAX base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

# vLLM needs this in order to work without error
ENV LD_PRELOAD=/usr/local/cuda/compat/libcuda.so

RUN apt-get update && apt-get --yes install curl git vim sudo unzip libssl-dev gcc rsync tmux
RUN pip install nvitop ipython fire pytest ruff evaluate scikit-learn
RUN cd /usr/src && git clone https://github.com/neuralmagic/AutoFP8.git && cd AutoFP8 && pip install -e .

ENTRYPOINT []
CMD []
