from pathlib import Path


def read_yaml_file(path: Path | str):
    from yaml import safe_load

    path = path if isinstance(path, Path) else Path(path)

    text_source = path.read_text(encoding="utf-8")
    data = safe_load(text_source)
    data["text_source"] = text_source
    return data
