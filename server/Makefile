include Makefile-flash-att
include Makefile-flash-att-v2
include Makefile-vllm

unit-tests:
	pytest -s -vv -m "not private" tests

gen-server:
	# Compile protos
	pip install grpcio-tools==1.51.1 mypy-protobuf==3.4.0 'types-protobuf>=3.20.4' --no-cache-dir
	mkdir lorax_server/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=lorax_server/pb \
		--grpc_python_out=lorax_server/pb --mypy_out=lorax_server/pb ../proto/generate.proto
	find lorax_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch lorax_server/pb/__init__.py

install-torch:
	# Install specific version of torch
	pip install torch --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

install: gen-server install-torch
	pip install pip --upgrade
	pip install -r requirements.txt
	pip install -e ".[bnb, accelerate]"

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 lorax_server/cli.py serve meta-llama/Llama-2-7b-hf --sharded
	# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 lorax_server/cli.py serve mistralai/Mistral-7B-Instruct-v0.1 --sharded
	# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 lorax_server/cli.py serve alexsherstinsky/Mistral-7B-v0.1-sharded --sharded

export-requirements:
	poetry export -o requirements.txt -E bnb -E quantize --without-hashes
