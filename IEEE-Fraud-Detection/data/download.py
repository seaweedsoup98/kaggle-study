"""Kaggle 규칙 동의와 API 인증 후 실행: python data/download.py."""

from loader import DATA_FILES, get_data_dir


if __name__ == "__main__":
    path = get_data_dir()
    print(f"data_dir: {path}")
    for name in DATA_FILES:
        print(f"{name}: {(path / name).stat().st_size / 1024**2:,.1f} MB")
