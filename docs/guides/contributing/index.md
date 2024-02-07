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
