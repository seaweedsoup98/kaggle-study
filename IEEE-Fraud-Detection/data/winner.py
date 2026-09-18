"""Winner's published V subset and released UID post-processing inputs."""
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd


V_KEEP = [
    1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30,
    36, 37, 40, 41, 44, 47, 48, 54, 56, 59, 62, 65, 67, 68, 70,
    76, 78, 80, 82, 86, 88, 89, 91, 107, 108, 111, 115, 117, 120, 121, 123,
    124, 127, 129, 130, 136, 138, 139, 142, 147, 156, 162, 165, 160, 166,
    178, 176, 173, 182, 187, 203, 205, 207, 215, 169, 171, 175, 180, 185,
    188, 198, 210, 209, 218, 223, 224, 226, 228, 229, 235,
    240, 258, 257, 253, 252, 260, 261, 264, 266, 267, 274, 277,
    220, 221, 234, 238, 250, 271, 294, 284, 285, 286, 291, 297,
    303, 305, 307, 309, 310, 320, 281, 283, 289, 296, 301, 314,
]
RELEASED_DATASET = "kyakovlev/ieee-submissions-and-uids/versions/2"
PP_FILES = ["uids_v4_no_multiuid_cleaning..csv", "uids_v1_no_multiuid_cleaning.csv"]


def get_released_dir():
    return Path(kagglehub.dataset_download(RELEASED_DATASET))


def postprocess(train_labels, predictions, released_dir):
    """XGB source cell 45: v4 then v1; retain unmatched rows and current train values."""
    combined = pd.concat([train_labels, predictions]).astype(float)
    assert combined.index.is_unique
    assert combined.between(0, 1).all()
    coverage = []
    for name in PP_FILES:
        uid = pd.read_csv(Path(released_dir) / name, usecols=["TransactionID", "uid"])
        assert uid.TransactionID.is_unique
        groups = uid.set_index("TransactionID").uid.reindex(combined.index)
        valid = groups.gt(0)
        means = combined.groupby(groups).mean()
        combined.loc[valid] = groups.loc[valid].map(means)
        coverage.append({"file": name, "test_coverage": float(valid.reindex(predictions.index).mean())})
    result = combined.reindex(predictions.index).rename("isFraud")
    assert np.isfinite(result).all() and result.between(0, 1).all()
    return result, coverage
