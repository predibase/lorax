def prepend(prefix: str, path: str) -> str:
    return f"{prefix}.{path}" if prefix else path
