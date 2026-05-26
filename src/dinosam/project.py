from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root() / path


def require_paths(paths: dict[str, str | Path]) -> bool:
    ok = True
    for name, path_value in paths.items():
        path = resolve_project_path(path_value)
        if path.exists():
            print(f"[OK] {name}: {path}")
            continue

        print(f"[MISSING] {name}: {path}")
        ok = False
    return ok
