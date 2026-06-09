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
