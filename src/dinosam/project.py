from pathlib import Path


def repo_root() -> Path:
    """返回当前 dinosam-lab 仓库的根目录路径。"""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    """把相对项目路径转换成基于仓库根目录的绝对路径。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root() / path


def require_paths(paths: dict[str, str | Path]) -> bool:
    """批量检查一组路径是否存在，并返回整体检查是否通过。"""
    ok = True
    for name, path_value in paths.items():
        path = resolve_project_path(path_value)
        if path.exists():
            print(f"[OK] {name}: {path}")
            continue

        print(f"[MISSING] {name}: {path}")
        ok = False
    return ok
