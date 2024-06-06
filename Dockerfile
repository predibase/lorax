# Final image
FROM ghcr.io/predibase/lorax:c71861a

RUN DEBIAN_FRONTEND=noninteractive apt install pkg-config rsync tmux rust-gdb git -y
RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP
RUN hash -r

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH=$PATH:$HOME/.cargo/bin

ENTRYPOINT ["lorax-launcher"]
CMD ["--json-output"]
