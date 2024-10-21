volume=$PWD/data

docker run \
    --name lorax-amd-dev \
    --cap-add=SYS_PTRACE \
    --device /dev/kfd \
    --device /dev/dri \
    --shm-size 1g \
    -v $volume:/data \
    -v /home/ubuntu/lorax:/lorax \
    -itd --entrypoint /bin/bash lorax-amd
