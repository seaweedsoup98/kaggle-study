"""데이터 경로 해석 및 로딩 유틸.

데이터 원본은 Kaggle community competition ``original-instant-gratification`` 이며,
용량이 커서(train/test 각 ~1.3GB) 저장소에는 포함하지 않는다.
아래 순서로 데이터 디렉터리를 찾는다.

1. 환경변수 ``IG_DATA_DIR``
2. ``data/raw`` (직접 복사해 둔 경우)
3. ``data/data_path.txt`` (``python data/download.py`` 가 기록해 둔 경로)
4. kagglehub 캐시 / 다운로드 (``download=True`` 인 경우)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

COMPETITION = "original-instant-gratification"
DATA_FILES = ("train.csv", "public_test.csv", "sample_submission.csv")

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
PATH_MEMO_FILE = PACKAGE_DIR / "data_path.txt"
LOCAL_RAW_DIR = PACKAGE_DIR / "raw"


def _is_valid(path: Path | None) -> bool:
    return path is not None and all((path / name).exists() for name in DATA_FILES)


def get_data_dir(data_dir: str | os.PathLike | None = None, download: bool = True) -> Path:
    """train/public_test/sample_submission 이 모두 들어있는 디렉터리를 반환한다."""
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    if os.environ.get("IG_DATA_DIR"):
        candidates.append(Path(os.environ["IG_DATA_DIR"]))
    candidates.append(LOCAL_RAW_DIR)
    if PATH_MEMO_FILE.exists():
        candidates.append(Path(PATH_MEMO_FILE.read_text(encoding="utf-8").strip()))

    for candidate in candidates:
        if _is_valid(candidate):
            return candidate

    if not download:
        raise FileNotFoundError(
            "데이터 디렉터리를 찾지 못했습니다. `python data/download.py` 를 먼저 실행하거나 "
            "IG_DATA_DIR 환경변수를 설정하세요."
        )

    import kagglehub  # 지연 import: 캐시된 경로가 있으면 필요 없음

    downloaded = Path(kagglehub.competition_download(COMPETITION))
    if not _is_valid(downloaded):
        raise FileNotFoundError(f"다운로드된 경로에 필요한 파일이 없습니다: {downloaded}")
    PATH_MEMO_FILE.write_text(str(downloaded), encoding="utf-8")
    return downloaded


def load_data(
    data_dir: str | os.PathLike | None = None,
    download: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train_df, test_df) 를 반환한다."""
    path = get_data_dir(data_dir, download=download)
    train_df = pd.read_csv(path / "train.csv")
    test_df = pd.read_csv(path / "public_test.csv")
    return train_df, test_df


def load_sample_submission(
    data_dir: str | os.PathLike | None = None,
    download: bool = True,
) -> pd.DataFrame:
    path = get_data_dir(data_dir, download=download)
    return pd.read_csv(path / "sample_submission.csv")
