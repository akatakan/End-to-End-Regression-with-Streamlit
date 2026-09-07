"""Notebook yontemi vs yeni yontem - ayni ortam, ayni split, ayni test satirlari.

Repodaki model_pipeline.pkl sklearn 1.5.0 ile uretildigi icin guncel sklearn'de
ne yukleniyor ne calisiyor. O yuzden notebook'un preprocessing zincirini ve
pipeline'ini birebir yeniden kuruyoruz (num_cols hatasi dahil) ve iki yontemi
ayni test satirlari uzerinde olcuyoruz.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train import RANDOM_STATE, clean_raw, evaluate, log_iqr_bounds, make_model
from features import RAW_COLUMNS

warnings.filterwarnings("ignore")


def notebook_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook'taki temizlik zinciri, hucre sirasiyla ve birebir."""
    df = df[~(df["Price"] < 10000)].copy()
    df.loc[(df["Floor"].isnull()) & (df["House Type"] != "Bina")
           & (df["House Type"] != "Daire"), "Floor"] = "Müstakil"
    df.loc[df["Bathroom Count"].isnull(), "Bathroom Count"] = 1
    df["House Age"] = df["House Age"].replace("Sıfır", 0)
    df.loc[df["House Age"] == 0, "Hand"] = "Sıfır"
    df["Heater Fuel"] = df["Heater Fuel"].replace("Kömür-Odun", "Kömür")
    df["Heater Type"] = df["Heater Type"].replace("Merkezi (Pay Ölçer)", "Merkezi")
    df["Floor"] = df["Floor"].apply(lambda x: "Bodrum" if isinstance(x, str) and "Bodrum" in x else x)
    df["Floor"] = df["Floor"].replace("En Üst Kat", "Çatı Katı")
    df["House Type"] = df["House Type"].replace("Yalı Dairesi", "Yalı")

    q1, q3 = df["Price"].quantile(0.25), df["Price"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df.loc[~((df["Price"] < lower) | (df["Price"] > upper))]

    df.loc[df["Heater Type"] == "Belirtilmemiş", "Heater Type"] = np.nan
    df = df[df["Heater Type"].notnull()]
    df["House Age"] = pd.to_numeric(df["House Age"], errors="coerce")
    df = df.loc[~((df["Price"] > 100000) & (df["House Type"] == "Daire") & (df["House Size"] < 100))]
    for col in ["Floor", "Heater Type", "Heater Fuel"]:
        vc = df[col].value_counts()
        df = df[df[col].isin(vc[vc >= 5].index)]
    df = df.loc[~(df["Price"] > 400000)]
    df[["Bedroom Count", "Hall Count"]] = df["Room Count"].str.split("+", expand=True).astype("int")
    df = df.drop("Room Count", axis=1)
    df = df.loc[~(df["Price"] < lower) | (df["Price"] > upper)]   # bozuk hucre, aynen
    df = df[~(df["Price"] > 40000)]
    return df


# Notebook'un `select_dtypes([...]).columns[:-1]` satiri ortama gore farkli
# sonuc veriyor:
#   * orijinal ortamda (numpy<2) Bedroom/Hall int32 idi -> num_cols'tan dustu,
#     [:-1] sadece Price'i atti. Repodaki model_pipeline.pkl bu haliyle uretilmis.
#   * guncel numpy'da Bedroom/Hall int64 -> [:-1] Hall Count'u atiyor ve
#     PRICE ozellik listesinde kaliyor; ColumnTransformer fit'te patliyor.
# Karsilastirmayi repodaki modelin gercekten kullandigi sutunlarla yapiyoruz.
NOTEBOOK_NUM_COLS = ["House Age", "House Size", "Bathroom Count"]


def notebook_pipeline(df: pd.DataFrame):
    """Notebook'un pipeline'i - num_cols[:-1] hatasinin sonucu dahil."""
    naive = df.select_dtypes(["float64", "int64"]).columns[:-1].tolist()
    print("  notebook'un num_cols ifadesi bu ortamda:", naive)
    num_cols = NOTEBOOK_NUM_COLS
    cat_cols = df.select_dtypes(["O", "object", "string", "str"]).columns
    num_tr = Pipeline([("i", SimpleImputer(strategy="mean")), ("s", StandardScaler())])
    cat_tr = Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                       ("e", OneHotEncoder(handle_unknown="ignore"))])
    ct = ColumnTransformer([("categorical", cat_tr, cat_cols), ("numerical", num_tr, num_cols)])
    pipe = Pipeline([("preprocessing", ct), ("model", CatBoostRegressor(verbose=0,
                                                                       allow_writing_files=False))])
    return pipe, list(num_cols), list(cat_cols)


def main() -> None:
    raw = pd.read_csv("emlak.csv").drop_duplicates()
    train_raw, test_raw = train_test_split(raw, test_size=0.2, random_state=RANDOM_STATE)
    print("ortak evren: %d satir | train %d | test %d"
          % (len(raw), len(train_raw), len(test_raw)))

    # --- YENI YONTEM ---
    tr_new = clean_raw(train_raw, verbose=False)
    lo, hi = log_iqr_bounds(tr_new["Price"])
    tr_new = tr_new[(tr_new["Price"] >= lo) & (tr_new["Price"] <= hi)]
    new = make_model()
    new.fit(tr_new[RAW_COLUMNS], tr_new["Price"])
    print("yeni yontem train: %d satir | fiyat araligi %s-%s TL"
          % (len(tr_new), f"{lo:,.0f}", f"{hi:,.0f}"))

    # --- NOTEBOOK YONTEMI ---
    tr_old = notebook_clean(train_raw)
    old, num_cols, cat_cols = notebook_pipeline(tr_old)
    y_old = tr_old["Price"]
    X_old = tr_old.drop("Price", axis=1)
    old.fit(X_old, y_old)
    print("notebook train: %d satir | fiyat araligi %s-%s"
          % (len(tr_old), f"{y_old.min():,.0f}", f"{y_old.max():,.0f}"))
    print("notebook num_cols:", num_cols)
    dropped = [c for c in X_old.columns if c not in num_cols and c not in list(cat_cols)]
    print("notebook'un HIC KULLANMADIGI sutunlar:", dropped)

    # --- ORTAK TEST ---
    band = (test_raw["Price"] >= 10000) & (test_raw["Price"] <= 40000)
    te = test_raw[band]
    print("\n" + "=" * 70)
    print("ORTAK TEST: 10.000-40.000 TL bandi, %d satir" % len(te))
    print("(notebook modeli bu bandin disinda tahmin uretemez)")
    print("=" * 70)

    evaluate(te["Price"], new.predict(te[RAW_COLUMNS]), "YENI  (CatBoost cat_features + log hedef)")

    te_old = te.copy()
    te_old[["Bedroom Count", "Hall Count"]] = (
        te_old["Room Count"].str.split("+", expand=True).astype("int"))
    te_old = te_old.drop("Room Count", axis=1)
    evaluate(te["Price"], old.predict(te_old[X_old.columns]), "ESKI  (notebook yontemi, ayni split)")

    # Ayni banda kisitlanmis yeni yontem: ozellik muhendisligi ve kategorik
    # islemenin katkisini kapsam farkindan ayirmak icin.
    tr_band = tr_new[(tr_new["Price"] >= 10000) & (tr_new["Price"] <= 40000)]
    new_band = make_model()
    new_band.fit(tr_band[RAW_COLUMNS], tr_band["Price"])
    evaluate(te["Price"], new_band.predict(te[RAW_COLUMNS]),
             "YENI  (ayni banda kisitli egitim, %d satir)" % len(tr_band))

    print("\n" + "=" * 70)
    print("YENI modelin tum test setindeki performansi (%d satir, 1.000-1.000.000 TL)" % len(test_raw))
    print("=" * 70)
    tn = clean_raw(test_raw, verbose=False)
    evaluate(tn["Price"], new.predict(tn[RAW_COLUMNS]), "YENI  (tum aralik)")
    print("\n  ESKI model bu satirlarin %%%.0f'ini zaten tahmin edemez (>40.000 TL)."
          % (100 * (tn["Price"] > 40000).mean()))


if __name__ == "__main__":
    main()
