"""원본 CSV 경로 해석과 TransactionID 기준 결합. 전처리는 노트북에서 한다."""

import os
from pathlib import Path

import pandas as pd

COMPETITION = "ieee-fraud-detection"
DATA_FILES = (
    "train_transaction.csv", "train_identity.csv",
    "test_transaction.csv", "test_identity.csv", "sample_submission.csv",
)
DATA_DIR = Path(__file__).resolve().parent
PATH_MEMO = DATA_DIR / "data_path.txt"


def get_data_dir(data_dir=None, download=True):
    """명시 경로 → IEEE_DATA_DIR → data/raw → 경로 메모 → 공유 캐시."""
    candidates = [data_dir, os.getenv("IEEE_DATA_DIR"), DATA_DIR / "raw"]
    if PATH_MEMO.exists():
        candidates.append(PATH_MEMO.read_text(encoding="utf-8").strip())
    candidates.append(Path.home() / ".cache/kagglehub/competitions" / COMPETITION)
    for candidate in candidates:
        if candidate and all((Path(candidate) / name).is_file() for name in DATA_FILES):
            return Path(candidate).resolve()
    if not download:
        raise FileNotFoundError("python data/download.py를 실행하거나 IEEE_DATA_DIR를 지정하세요.")
    import kagglehub
    path = Path(kagglehub.competition_download(COMPETITION))
    if not all((path / name).is_file() for name in DATA_FILES):
        raise FileNotFoundError(f"원본 CSV 5개가 필요합니다: {path}")
    PATH_MEMO.write_text(str(path), encoding="utf-8")
    return path


def load_data(data_dir=None, nrows=None):
    """(train, test). nrows는 빠른 실행용이며 identity는 해당 ID만 결합한다."""
    path = get_data_dir(data_dir)
    frames = []
    for split in ("train", "test"):
        transactions = pd.read_csv(path / f"{split}_transaction.csv", nrows=nrows)
        identity = pd.read_csv(path / f"{split}_identity.csv")
        identity.columns = identity.columns.str.replace("-", "_", regex=False)
        identity = identity.loc[identity.TransactionID.isin(transactions.TransactionID)]
        frames.append(transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one"))
    return tuple(frames)


def load_sample_submission(data_dir=None):
    return pd.read_csv(get_data_dir(data_dir) / "sample_submission.csv")
