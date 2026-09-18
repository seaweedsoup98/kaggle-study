"""Port of kyakovlev's public minification → ieee-fe-with-some-eda pipeline.

Keep float32 instead of float16; defer ProductCD/M4 target means to each CV fold.
No dependency on externally prepared pickles. Model code stays in notebooks.
"""
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.tseries.holiday import USFederalHolidayCalendar

from data.loader import get_data_dir

CAT_ORIGINAL = ["ProductCD", "M4", "card1", "card2", "card3", "card4", "card5", "card6",
                "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]


def build_features(nrows=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _build_features(nrows)


def _build_features(nrows=None):
    raw = get_data_dir()
    frames, identities = [], []
    for split in ["train", "test"]:
        frame = pd.read_csv(raw / f"{split}_transaction.csv")
        if split == "test":
            frame["isFraud"] = 0
        if nrows:
            dates = pd.Timestamp("2017-11-30") + pd.to_timedelta(frame.TransactionDT, unit="s")
            frame = frame.groupby(dates.dt.to_period("M"), group_keys=False).sample(
                frac=min(1, nrows / len(frame)), random_state=42).sort_index().reset_index(drop=True)
        identity = pd.read_csv(raw / f"{split}_identity.csv")
        identity.columns = identity.columns.str.replace("-", "_", regex=False)
        identity = identity[identity.TransactionID.isin(frame.TransactionID)].reset_index(drop=True)
        for data in [frame, identity]:
            for col in data.select_dtypes("float64"):
                data[col] = data[col].astype("float32")
        frames.append(frame.copy())
        identities.append(identity.copy())
    train, test = frames

    def frequency(columns, self_encoding=False):
        for col in columns:
            counts = pd.concat([train[col], test[col]]).value_counts(dropna=False)
            for frame in frames:
                frame[col if self_encoding else col + "_fq_enc"] = frame[col].map(counts)

    # Source minification: four frequency-coded categories; M/identity mappings.
    for col in ["card4", "card6", "ProductCD", "M4"]:
        counts = pd.concat([train[col], test[col]]).value_counts()
        for frame in frames:
            frame[col] = frame[col].map(counts)
    for frame in frames:
        for col in ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]:
            frame[col] = frame[col].map({"T": 1, "F": 0})
    original = pd.concat([frame[["TransactionID"] + CAT_ORIGINAL] for frame in frames]).set_index("TransactionID")
    identity_maps = {"id_12": {"Found": 1, "NotFound": 0}, "id_15": {"New": 2, "Found": 1, "Unknown": 0},
                     "id_16": {"Found": 1, "NotFound": 0}, "id_23": {"TRANSPARENT": 4, "IP_PROXY": 3, "IP_PROXY:ANONYMOUS": 2, "IP_PROXY:HIDDEN": 1},
                     "id_27": {"Found": 1, "NotFound": 0}, "id_28": {"New": 2, "Found": 1}, "id_29": {"Found": 1, "NotFound": 0}}
    identity_maps.update({f"id_{i}": {"T": 1, "F": 0} for i in [35, 36, 37, 38]})
    for identity in identities:
        for col, mapping in identity_maps.items():
            identity[col] = identity[col].map(mapping)
        identity["id_34"] = pd.to_numeric(identity.id_34.fillna(":0").str.split(":").str[1]).replace(0, np.nan)
        resolution = identity.id_33.fillna("0x0").str.split("x", expand=True).astype(int)
        identity["id_33_0"], identity["id_33_1"] = resolution[0], resolution[1]
    values, _ = pd.concat([i.id_33.fillna("unseen_before_label") for i in identities]).factorize(sort=True)
    identities[0]["id_33"], identities[1]["id_33"] = values[:len(identities[0])], values[len(identities[0]):]

    remove = ["TransactionID", "TransactionDT", "isFraud"]
    holidays = USFederalHolidayCalendar().holidays("2017-10-01", "2019-01-01")
    for frame in frames:
        dates = pd.Timestamp("2017-11-30") + pd.to_timedelta(frame.TransactionDT, unit="s")
        frame["DT_M"] = (dates.dt.year - 2017) * 12 + dates.dt.month
        frame["DT_W"] = (dates.dt.year - 2017) * 52 + dates.dt.isocalendar().week.astype(int)
        frame["DT_D"] = (dates.dt.year - 2017) * 365 + dates.dt.dayofyear
        frame["DT_hour"], frame["DT_day_week"], frame["DT_day_month"] = dates.dt.hour, dates.dt.dayofweek, dates.dt.day
        frame["is_december"], frame["is_holiday"] = dates.dt.month.eq(12).astype("int8"), dates.dt.normalize().isin(holidays).astype("int8")
    remove += ["DT_M", "DT_W", "DT_D", "DT_hour", "DT_day_week", "DT_day_month"]
    for period in ["DT_M", "DT_W", "DT_D"]:
        counts = pd.concat([train[period], test[period]]).value_counts()
        for frame in frames:
            frame[period + "_total"] = frame[period].map(counts)
        remove.append(period + "_total")

    # The source deliberately drops rare cards and nonintersecting card categories.
    for col in [f"card{i}" for i in range(1, 7)]:
        counts = pd.concat([train[col], test[col]]).value_counts()
        train[col] = train[col].where(train[col].isin(test[col]))
        test[col] = test[col].where(test[col].isin(train[col]))
        if col == "card1":
            for frame in frames:
                frame[col] = frame[col].where(frame[col].isin(counts[counts > 2].index))
    for frame in frames:
        frame["uid"] = frame.card1.astype(str).fillna("nan") + "_" + frame.card2.astype(str).fillna("nan")
        frame["uid2"] = frame.uid + "_" + frame.card3.astype(str).fillna("nan") + "_" + frame.card5.astype(str).fillna("nan")
        frame["uid3"] = frame.uid2 + "_" + frame.addr1.astype(str).fillna("nan") + "_" + frame.addr2.astype(str).fillna("nan")
        frame["uid4"], frame["uid5"] = frame.uid3 + "_" + frame.P_emaildomain.astype(str).fillna("nan"), frame.uid3 + "_" + frame.R_emaildomain.astype(str).fillna("nan")
        frame["bank_type"] = frame.card3.astype(str).fillna("nan") + "_" + frame.card5.astype(str).fillna("nan")
    uids = ["uid", "uid2", "uid3", "uid4", "uid5", "bank_type"]
    remove += uids
    frequency(["card1", "card2", "card3", "card5"] + uids[:5])
    for frame in frames:
        for col in ["card3", "card5", "bank_type"]:
            for period, value, label in [("DT_D", "DT_hour", "hour"), ("DT_W", "DT_day_week", "week_day"), ("DT_M", "DT_day_month", "month_day")]:
                key = frame[col].astype(str).fillna("nan") + "_" + frame[period].astype(str).fillna("nan")
                frame[f"{col}_{period}_{label}_dist"] = frame[value] - frame.groupby(key)[value].transform("mean")
                modes = frame.groupby([col, period, value]).size().reset_index(name="count").sort_values([col, period, "count"])
                modes = modes.drop_duplicates([col, period], keep="last")
                mapping = pd.Series(modes[value].to_numpy(), index=modes[col].astype(str).fillna("nan") + "_" + modes[period].astype(str).fillna("nan"))
                frame[f"{col}_{period}_{label}_dist_best"] = frame[value] - key.map(mapping)

    def time_frequency(columns):
        for period in ["DT_M", "DT_W", "DT_D"]:
            for col in columns:
                keys = [frame[col].astype(str).fillna("nan") + "_" + frame[period].astype(str).fillna("nan") for frame in frames]
                counts = pd.concat(keys).value_counts()
                for frame, key in zip(frames, keys):
                    frame[col + "_" + period] = key.map(counts) / frame[period + "_total"]

    def aggregate(columns, groups):
        for group in groups:
            for col in columns:
                combined = pd.concat([train[[group, col]], test[[group, col]]])
                for stat in ["mean", "std"]:
                    mapping = combined.groupby(group)[col].agg(stat)
                    for frame in frames:
                        frame[f"{group}_{col}_{stat}"] = frame[group].map(mapping)

    def normalize(columns):
        for frame in frames:
            for period in ["DT_D", "DT_W", "DT_M"]:
                for col in columns:
                    group = frame.groupby(period)[col]
                    minimum, maximum = group.transform("min"), group.transform("max")
                    frame[f"{col}_{period}_min_max"] = (frame[col] - minimum) / (maximum - minimum)
                    frame[f"{col}_{period}_std_score"] = (frame[col] - group.transform("mean")) / group.transform("std")

    time_frequency(["bank_type"])
    ds = [f"D{i}" for i in range(1, 16)]
    aggregate(ds, uids)
    for frame in frames:
        frame[ds] = frame[ds].clip(lower=0)
        frame["D9_not_na"], frame["D8_not_same_day"] = frame.D9.notna().astype(int), frame.D8.ge(1).astype(int)
        frame["D8_D9_decimal_dist"] = (frame.D8.fillna(0) - frame.D8.fillna(0).astype(int) - frame.D9).abs()
        frame["D8"] = frame.D8.fillna(-1).astype(int)
    normalize([c for c in ds if c not in ["D1", "D2", "D9"]])
    for col in ["D1", "D2"]:
        maximum = train[col].max()
        for frame in frames:
            frame[col + "_scaled"] = frame[col] / maximum
    frequency(ds, self_encoding=True)
    frames = [frame.copy() for frame in frames]
    train, test = frames
    for frame in frames:
        frame["TransactionAmt"] = frame.TransactionAmt.clip(0, 5000)
    train["TransactionAmt_check"] = train.TransactionAmt.isin(test.TransactionAmt).astype(int)
    test["TransactionAmt_check"] = test.TransactionAmt.isin(train.TransactionAmt).astype(int)
    aggregate(["TransactionAmt"], ["card1", "card2", "card3", "card5"] + uids)
    normalize(["TransactionAmt"])
    for frame in frames:
        frame["product_type"] = frame.ProductCD.astype(str).fillna("nan") + "_" + frame.TransactionAmt.astype(str).fillna("nan")
    time_frequency(["product_type"])
    frequency(["product_type"], self_encoding=True)
    cs = [f"C{i}" for i in range(1, 15)]
    frequency(cs)
    for col in cs:
        maximum = train.loc[train.DT_M.eq(train.DT_M.max()), col].max()
        for frame in frames:
            frame[col] = frame[col].clip(upper=maximum)
    for frame in frames:
        frame["TransactionAmt"] = np.log1p(frame.TransactionAmt)
    for identity in identities:
        for col in ["DeviceInfo", "id_30", "id_31"]:
            identity[col] = identity[col].fillna("unknown_device").str.lower()
            suffix = "device"
            identity[col + "_" + suffix] = identity[col].map(lambda s: "".join(c for c in s if c.isalpha()))
            if col != "id_31":
                identity[col + "_version"] = identity[col].map(lambda s: "".join(c for c in s if c.isnumeric()))
    frames = [frame.merge(identity, on="TransactionID", how="left", validate="one_to_one") for frame, identity in zip(frames, identities)]
    train, test = frames
    frequency(["DeviceInfo", "DeviceInfo_device", "DeviceInfo_version", "id_30", "id_30_device", "id_30_version", "id_31", "id_31_device", "id_33"], self_encoding=True)
    # Source global target means are intentionally delayed until model CV.
    for col in train:
        if is_string_dtype(train[col]):
            values, _ = pd.concat([train[col].fillna("unseen_before_label"), test[col].fillna("unseen_before_label")]).factorize(sort=True)
            train[col] = pd.Categorical(values[:len(train)])
            test[col] = pd.Categorical(values[len(train):])
            categories = sorted(set(train[col].cat.categories) | set(test[col].cat.categories))
            train[col], test[col] = train[col].cat.set_categories(categories), test[col].cat.set_categories(categories)
    for frame in frames:
        for col in frame.select_dtypes("float64"):
            frame[col] = frame[col].astype("float32")
    return train.copy(), test.copy(), remove, original
