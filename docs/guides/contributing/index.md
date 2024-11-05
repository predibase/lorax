# Contributing to LoRAX

## Setting up your development environment

See [Development Environment](./development_env.md).

## Updating Python server dependencies

LoRAX uses [Poetry](https://python-poetry.org/) to manage dependencies.

When modifying the dependencies of the LoRAX Python server, first modify the server [pyproject.toml](https://github.com/predibase/lorax/blob/main/server/pyproject.toml) file directly making the desired changes.

Next, from within the `server` directory, generate an updated `poetry.lock` file:

```shell
poetry lock --no-update 
```

Then (still within the `server` directory) generate a new `requirements.txt` file:

```shell
make export-requirements
```

Never modify `requirements.txt` directly, as it may introduce dependency conflicts.

## Profiling

LoRAX supports the [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) to measure performance of LoRAX.

You can enable profiling when launching LoRAX by setting the `LORAX_PROFILER_DIR` environment variable to the directory
you wish to output the Tensorboard traces to.

Once initialized, LoRAX will begin recording traces for every request to the server. Because traces can get very large,
we record only the first 10 prefill requests (plus any decode requests between them), then stop recording and write
out the results. A summary will be printed to stdout when this occurs.

Once you have your traces written to the profiler directory, you can visualize them in Tensorboard using the
[PyTorch Profiler Tensorboard Plugin](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).

```bash
pip install torch_tb_profiler
tensorboard --logdir=$LORAX_PROFILER_DIR
```
