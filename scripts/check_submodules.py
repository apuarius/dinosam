from pathlib import Path


def check_path(name: str, path: Path) -> bool:
    if path.exists():
        print(f"[OK] {name}: {path}")
        return True

    print(f"[MISSING] {name}: {path}")
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    checks = {
        "DINOv3": repo_root / "third_party" / "dinov3",
        "SAM2": repo_root / "third_party" / "sam2",
        ".gitmodules": repo_root / ".gitmodules",
    }

    ok = True
    for name, path in checks.items():
        ok = check_path(name, path) and ok

    if not ok:
        print()
        print("Run this command to initialize submodules:")
        print("git submodule update --init --recursive")
        return 1

    print()
    print("All submodule paths look good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
