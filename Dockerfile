FROM ghcr.io/predibase/lorax:3305da5

CMD apt-get update && apt-get --yes install curl git vim sudo unzip libssl-dev gcc rsync tmux
CMD pip install nvitop ipython fire pytest ruff evaluate scikit-learn
CMD cd /usr/src && git clone https://github.com/neuralmagic/AutoFP8.git && cd AutoFP8 && pip install -e .

ENTRYPOINT []
CMD []
