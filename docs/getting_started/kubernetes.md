# Kubernetes (Helm)

LoRAX includes Helm charts that make it easy to start using LoRAX in production with high availability and load balancing on Kubernetes.

To spin up a LoRAX deployment with Helm, you only need to be connected to a Kubernetes cluster through `kubectl``. We provide a default values.yaml file that can be used to deploy a Mistral 7B base model to your Kubernetes cluster:

```shell
helm install mistral-7b-release charts/lorax
```

The default [values.yaml](https://github.com/predibase/lorax/blob/main/charts/lorax/values.yaml) configuration deploys a single replica of the Mistral 7B model. You can tailor configuration parameters to deploy any Llama or Mistral model by creating a new values file from the template and updating variables. Once a new values file is created, you can run the following command to deploy your LLM with LoRAX:

```shell
helm install -f your-values-file.yaml your-model-release charts/lorax
```

To delete the resources:

```shell
helm uninstall your-model-release
```
