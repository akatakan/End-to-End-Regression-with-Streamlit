"""Ozellik uretimi ve model sinifi.

Bu kod train.py'den ayri bir modulde duruyor cunku egitilmis pipeline
pickle'lanirken FunctionTransformer'in fonksiyonuna ve model sinifina
*import yoluyla* referans veriyor. train.py'nin icinde kalsalardi
pickle'da "__main__.build_features" yazardi ve app.py modeli yukleyemezdi.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

MISSING = "__missing__"

# app.py'nin gonderdigi ham sutunlar
RAW_COLUMNS = [
    "City", "Town", "Neighborhood", "House Type", "House Age", "House Size",
    "Room Count", "Floor", "Furniture", "Bathroom Count", "Hand",
    "Heater Type", "Heater Fuel",
]

CAT_FEATURES = [
    "City", "Town", "Neighborhood", "House Type", "Floor Type",
    "Furniture", "Hand", "Heater Type", "Heater Fuel",
]
NUM_FEATURES = [
    "House Age", "House Size", "Bathroom Count",
    "Bedroom Count", "Hall Count", "Room Total", "Floor Num", "Size Per Room",
]

_FLOOR_NUM = re.compile(r"^(\d+)\.\s*Kat$")
_FLOOR_SPECIAL = {
    "Giriş Katı": 0, "Zemin": 0, "Bahçe Katı": 0, "Yüksek Giriş": 0,
    "Bodrum ve Zemin": 0, "Müstakil": 0, "Villa Katı": 0,
    "Kot 1": -1, "Kot 2": -2, "Kot 3": -3,
    "Bodrum": -1, "Yarı Bodrum": -1,
    "21 ve üzeri": 21,
}
# bool / "Evet" / "Hayir" hepsi ayni etikete dusuyor -> app.py ile egitim
# arasindaki uyusmazlik burada kapaniyor
_FURNITURE_MAP = {
    True: "Esyali", "True": "Esyali", "Evet": "Esyali", "Eşyalı": "Esyali",
    False: "Esyasiz", "False": "Esyasiz", "Hayır": "Esyasiz", "Eşyasız": "Esyasiz",
}


def _floor_num(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    m = _FLOOR_NUM.match(value.strip())
    if m:
        return float(m.group(1))
    return float(_FLOOR_SPECIAL.get(value.strip(), np.nan))


def _as_category(series: pd.Series) -> pd.Series:
    """NaN'i acik bir etikete cevirip duz object dtype dondurur.

    CatBoost kategorik sutunlarda NaN kabul etmiyor; ayrica pandas 3.x'te
    .astype(str) StringDtype uretiyor ve CatBoost onu sayisal saniyor.
    """
    filled = series.astype("object").where(series.notna(), MISSING)
    return filled.map(str).astype("object")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ham ilan tablosu -> modele giren tablo. Satir silmez, saf donusum."""
    out = pd.DataFrame(index=df.index)

    for col in ["City", "Town", "Neighborhood", "House Type",
                "Hand", "Heater Type", "Heater Fuel"]:
        out[col] = _as_category(df[col])

    out["Furniture"] = _as_category(df["Furniture"].map(_FURNITURE_MAP))

    floor = df["Floor"]
    out["Floor Type"] = _as_category(floor)
    out["Floor Num"] = floor.map(_floor_num).astype(float)

    rooms = df["Room Count"].astype(str).str.extract(r"^\s*(\d+)\s*\+\s*(\d+)\s*$")
    out["Bedroom Count"] = pd.to_numeric(rooms[0], errors="coerce")
    out["Hall Count"] = pd.to_numeric(rooms[1], errors="coerce")
    out["Room Total"] = out["Bedroom Count"] + out["Hall Count"]

    out["House Age"] = pd.to_numeric(df["House Age"], errors="coerce")
    size = pd.to_numeric(df["House Size"], errors="coerce")
    # "1.100 m2" -> 1.1 seklinde bozulan degerleri tahminde de duzelt
    out["House Size"] = size.where(size >= 20, size * 1000)
    out["Bathroom Count"] = pd.to_numeric(df["Bathroom Count"], errors="coerce")
    out["Size Per Room"] = out["House Size"] / out["Room Total"].replace(0, np.nan)

    return out[CAT_FEATURES + NUM_FEATURES]


class CatBoostDF(CatBoostRegressor):
    """cat_features'i fit sirasinda DataFrame'den tespit eder.

    CatBoostRegressor'a cat_features'i constructor'da vermek sklearn'un
    clone()'unu bozuyor ("constructor either does not set or modifies
    parameter cat_features") -> cross_val_score/GridSearchCV calismiyor.
    """

    def fit(self, X, y=None, **kwargs):  # type: ignore[override]
        if isinstance(X, pd.DataFrame) and "cat_features" not in kwargs:
            kwargs["cat_features"] = [
                c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
            ]
        return super().fit(X, y, **kwargs)
