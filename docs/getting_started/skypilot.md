[SkyPilot](https://github.com/skypilot-org/skypilot) is a framework for running AI workloads
in the cloud of your choice (AWS, Azure, GCP, etc.). It abstracts away the complexity of finding available GPU resources across clouds / zones, syncing data between
storage systems, and managing the excution of distributed workloads.

## Setup

First install SkyPilot and check that your cloud credentials are properly set:

```shell
pip install skypilot
sky check
```

## Launch a deployment

Create a YAML configuration file called `lorax.yaml`:

```yaml
resources:
  cloud: aws
  accelerators: A10G:1
  memory: 32+
  ports: 
    - 8080

envs:
  MODEL_ID: mistralai/Mistral-7B-Instruct-v0.1

run: |
  docker run --gpus all --shm-size 1g -p 8080:80 -v ~/data:/data \
    ghcr.io/predibase/lorax:main \
    --model-id $MODEL_ID
```

In the above example, we're asking SkyPilot to provision an AWS instance with 1 Nvidia A10G GPU and at least 32GB of RAM. Once the node is provisioned,
SkyPilot will launch the LoRAX server using our latest pre-built [Docker image](./docker.md).

Let's launch our LoRAX job:

```shell
sky launch -c lorax-cluster lorax.yaml
```

By default, this config will deploy Mistral-7B-Instruct, but this can be overridden by running `sky launch` with the argument `--env MODEL_ID=<my_model>`.

!!! warn
    
    This config will launch the instance on a public IP. It's highly recommended to secure the instance within a private subnet. See the [Advanced Configurations](https://skypilot.readthedocs.io/en/latest/reference/config.html#config-yaml) section of the SkyPilot docs for options to run within VPC and setup private IPs.

## Prompt LoRAX

In a separate window, obtain the IP address of the newly created instance:

```shell
sky status --ip lorax-cluster
```

Now we can prompt the LoRAX deployment as usual:

```shell
IP=$(sky status --ip lorax-cluster)

curl http://$IP:8080/generate \
    -X POST \
    -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64, "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"}}' \
    -H 'Content-Type: application/json'
```

## Stop the deployment

Stopping the deployment will shut down the instance, but keep the storage volume:

```shell
sky stop lorax-cluster
```

Because we set `docker run ... -v ~/data:/data` in our config from before, this means any model weights or adapters we downloaded will be persisted the next time we run `sky launch`. The LoRAX Docker image will also be cached, meaning tags like `latest` won't be updated on restart unless you add `docker pull` to your `run` configuration.

## Delete the deployment

To completely delete the deployment, including the storage volume:

```shell
sky down lorax-cluster
```

The next time you run `sky launch`, the deployment will be recreated from scratch.
