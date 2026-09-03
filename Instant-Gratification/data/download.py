"""Kaggle competition 데이터를 내려받고 경로를 기록한다.

사전 준비
---------
1. https://www.kaggle.com/competitions/original-instant-gratification 에서 대회 참가
2. https://www.kaggle.com/settings/api 에서 `Generate New Token`
3. 내려받은 kaggle.json 을 `~/.kaggle/` 에 두거나 KAGGLE_USERNAME/KAGGLE_KEY 환경변수 설정

사용
----
    python data/download.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import COMPETITION, DATA_FILES, PATH_MEMO_FILE, get_data_dir  # noqa: E402


def main() -> None:
    print(f"[download] competition={COMPETITION}")
    path = get_data_dir(download=True)
    PATH_MEMO_FILE.write_text(str(path), encoding="utf-8")
    print(f"[download] data_dir={path}")
    for name in DATA_FILES:
        size_mb = (path / name).stat().st_size / 1024 / 1024
        print(f"  - {name}: {size_mb:,.1f} MB")
    print(f"[download] wrote resolved path to {PATH_MEMO_FILE.name}")


if __name__ == "__main__":
    main()
