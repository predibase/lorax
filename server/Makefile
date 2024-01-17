include Makefile-flash-att
include Makefile-flash-att-v2
include Makefile-vllm
include Makefile-megablocks

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

install: gen-server
	pip install pip --upgrade
	pip install torch==2.1.2
	pip install -r requirements.txt
	pip install -e ".[bnb, accelerate, quantize, peft]"

run-dev:
	# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 lorax_server/cli.py serve gpt2
	# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 lorax_server/cli.py serve meta-llama/Llama-2-7b-hf --sharded
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 lorax_server/cli.py serve mistralai/Mistral-7B-Instruct-v0.1 --sharded
	# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 lorax_server/cli.py serve mistralai/Mistral-7B-v0.1 --sharded
	# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 lorax_server/cli.py serve TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ --quantize awq --self-extend-attention

export-requirements:
	poetry export -o requirements.txt -E bnb --without-hashes
