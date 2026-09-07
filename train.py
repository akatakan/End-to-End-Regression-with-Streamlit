"""
Kiralik ev fiyat tahmini - egitim scripti (notebook'un yerine gecer).

Onemli farklar:
  * veri kalitesi duzeltmeleri (binlik ayirici, para birimi, duplike) gercekten yapiliyor
  * aykiri deger LOG ekseninde ve SADECE train setinden hesaplanan sinirlarla kesiliyor
  * hedef log1p uzerinde ogreniliyor, metrikler ters donusumden sonra TL cinsinden raporlaniyor
  * kategorikler CatBoost'un kendi cat_features'i ile isleniyor (1900 sutunluk OHE yok)
  * ozellik uretimi pipeline'in ICINDE -> app.py ham girdiyi verip predict cagirabiliyor

Kullanim:
    python train.py --data emlak.csv --out model_pipeline.pkl
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import (
    CAT_FEATURES,
    NUM_FEATURES,
    RAW_COLUMNS,
    CatBoostDF,
    build_features,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1) Veri kalitesi - sadece egitim verisine uygulanir (satir silme icerir)
# ---------------------------------------------------------------------------

# hepsiemlak fiyatlari TL disi para birimlerinde de listeleyebiliyor; scraper
# sembolu dusurdugu icin bu satirlar TL gibi gorunuyor. K.K.T.C. ilanlari
# GBP/EUR cinsinden ve medyanlari 925 (Turkiye medyani 20.000).
NON_TRY_CITIES = {"K.K.T.C."}

PRICE_FLOOR = 1_000        # altindakiler tuzak/placeholder ilan (1, 99, 199 TL)
PRICE_CEILING = 1_000_000  # ustundekiler yillik kira veya satilik karisimi
MAX_ROOMS = 10             # 25+2, 41+1, 18+18 -> yurt/bina ilani
MAX_AGE = 100


def clean_raw(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Hedeften bagimsiz veri kalitesi duzeltmeleri. Split'ten ONCE calisir."""
    log: list[tuple[str, int, int]] = []
    n0 = len(df)

    # Duplike kontrolu ozellik sutunlari uzerinden yapiliyor: yeni kaziyici
    # CSV'ye bir de "url" yaziyor ve tum sutunlara bakan bir drop_duplicates
    # hicbir tekrar bulamazdi.
    dedup_on = [c for c in RAW_COLUMNS + ["Price"] if c in df.columns]
    n = int(df.duplicated(subset=dedup_on).sum())
    df = df.drop_duplicates(subset=dedup_on).copy()
    log.append(("tam duplike", -n, len(df)))

    # "1.100 m2" -> float("1.100") == 1.1 ; 1000 m2 ustu her ev bozulmus
    bad_size = df["House Size"] < 20
    n = int(bad_size.sum())
    df.loc[bad_size, "House Size"] = df.loc[bad_size, "House Size"] * 1000
    log.append(("House Size binlik ayirici duzeltildi", n, len(df)))

    def drop(mask: pd.Series, why: str) -> None:
        nonlocal df
        k = int(mask.sum())
        if k:
            df = df[~mask].copy()
        log.append((why, -k, len(df)))

    # Yeni kaziyici para birimini kaydediyor; varsa ona guven, yoksa eski
    # CSV'ler icin sehir bazli yedek kurala dus.
    if "Currency" in df.columns:
        drop(df["Currency"].notna() & (df["Currency"] != "TRY"), "TL disi para birimi")
    else:
        drop(df["City"].isin(NON_TRY_CITIES), "TL disi para birimi (sehir)")
    drop(df["Price"] < PRICE_FLOOR, "Price < %d" % PRICE_FLOOR)
    drop(df["Price"] > PRICE_CEILING, "Price > %d" % PRICE_CEILING)
    drop(df["House Age"] > MAX_AGE, "House Age > %d" % MAX_AGE)

    rooms = pd.to_numeric(df["Room Count"].astype(str).str.split("+").str[0], errors="coerce")
    drop(rooms > MAX_ROOMS, "oda sayisi > %d" % MAX_ROOMS)

    # "Yapinin Durumu" alani bazi ilanlarda yapi turunu veriyor -> yanlis eslesme
    n = int((df["Hand"] == "Betonarme").sum())
    df.loc[df["Hand"] == "Betonarme", "Hand"] = np.nan
    log.append(("Hand == Betonarme -> NaN", n, len(df)))

    if verbose:
        print("\n--- VERI KALITESI  (%d satir) ---" % n0)
        for why, delta, left in log:
            print("  %-42s %+6d  ->  %d" % (why, delta, left))
        print("  %-42s %+6d  ->  %d  (%%%.1f atildi)"
              % ("SONUC", len(df) - n0, len(df), 100 * (n0 - len(df)) / n0))
    return df


def log_iqr_bounds(y: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Fiyat log-normal; IQR'i log ekseninde hesaplamak dogru olani.

    Ham IQR bu dagilimda alt siniri negatife itip gercek copu (1 TL, 99 TL)
    kacirirken ust taraftan mesru pahali ilanlari biciyor.
    """
    logy = np.log1p(y)
    q1, q3 = logy.quantile(0.25), logy.quantile(0.75)
    iqr = q3 - q1
    return float(np.expm1(q1 - k * iqr)), float(np.expm1(q3 + k * iqr))


# ---------------------------------------------------------------------------
# 2) Model
# ---------------------------------------------------------------------------

def make_model(**catboost_kwargs: object) -> TransformedTargetRegressor:
    """Ham girdi -> TL tahmini. Hedef log1p uzerinde ogreniliyor."""
    params: dict[str, object] = dict(
        loss_function="RMSE",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )
    params.update(catboost_kwargs)
    pipe = Pipeline([
        ("features", FunctionTransformer(build_features, validate=False)),
        ("model", CatBoostDF(**params)),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


# ---------------------------------------------------------------------------
# 3) Degerlendirme
# ---------------------------------------------------------------------------

def evaluate(y_true: object, y_pred: object, label: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = {
        "n": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "MedAPE%": float(np.median(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
        "log-RMSE": float(np.sqrt(mean_squared_error(
            np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))))),
        "within20%": float(np.mean(np.abs(y_pred / y_true - 1) <= 0.20) * 100),
    }
    print("\n  %s  (n=%d)" % (label, m["n"]))
    print("    RMSE   %10s TL      MAE      %10s TL" % (f"{m['RMSE']:,.0f}", f"{m['MAE']:,.0f}"))
    print("    MAPE   %10.1f %%       MedAPE   %10.1f %%" % (m["MAPE%"], m["MedAPE%"]))
    print("    R2     %10.3f         log-RMSE %10.3f" % (m["R2"], m["log-RMSE"]))
    print("    tahminlerin %%%.1f'i gercek degerin +-%%20 bandinda" % m["within20%"])
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="emlak.csv")
    ap.add_argument("--out", default="model_pipeline.pkl")
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--locations", default="locations.json")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-cv", action="store_true")
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    df = clean_raw(raw)

    X_all = df[RAW_COLUMNS]
    y_all = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=RANDOM_STATE
    )

    # Aykiri kirpma SADECE train'de. Test seti dokunulmadan kaliyor ki
    # raporlanan skor gercek dunyada gorulecek dagilimi temsil etsin.
    lo, hi = log_iqr_bounds(y_train)
    keep = (y_train >= lo) & (y_train <= hi)
    print("\n--- AYKIRI DEGER (train, log-IQR) ---")
    print("  sinirlar: %s .. %s TL" % (f"{lo:,.0f}", f"{hi:,.0f}"))
    print("  train %d -> %d (-%d)" % (len(y_train), int(keep.sum()), int((~keep).sum())))
    print("  test  %d (dokunulmadi)" % len(y_test))
    X_train, y_train = X_train[keep], y_train[keep]

    model = make_model()

    cv_summary = None
    if not args.no_cv:
        print("\n--- CAPRAZ DOGRULAMA (train, 5-fold) ---")
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = -cross_val_score(
            model, X_train, y_train, cv=kf,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        for i, s in enumerate(scores, 1):
            print("  fold %d: RMSE %s TL" % (i, f"{s:,.0f}"))
        print("  ortalama %s TL  |  std %s" % (f"{scores.mean():,.0f}", f"{scores.std():,.0f}"))
        cv_summary = {"mean_rmse": float(scores.mean()), "std_rmse": float(scores.std())}

    print("\n--- EGITIM ---")
    model.fit(X_train, y_train)

    print("\n--- TEST SETI ---")
    results = {}
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    results["baseline_median"] = evaluate(y_test, baseline.predict(X_test), "baseline (medyan)")
    results["catboost_full_test"] = evaluate(y_test, model.predict(X_test), "CatBoost - tum test")

    in_range = (y_test >= lo) & (y_test <= hi)
    results["catboost_in_range"] = evaluate(
        y_test[in_range], model.predict(X_test[in_range]),
        "CatBoost - train dagilimi icindeki test satirlari",
    )

    print("\n--- OZELLIK ONEMI ---")
    cb = model.regressor_.named_steps["model"]
    imp = (pd.Series(cb.get_feature_importance(), index=CAT_FEATURES + NUM_FEATURES)
             .sort_values(ascending=False))
    for name, val in imp.items():
        print("  %-16s %6.2f" % (name, val))

    # app.py'nin selectbox'lari bu listeden uretiliyor -> arayuz ile modelin
    # gordugu kategoriler bir daha ayrisamaz (eski app.py'de elle yazilmis
    # listeler modelin kategorileriyle uyusmuyordu).
    ui_choices = {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in ["House Type", "Room Count", "Floor", "Heater Type", "Heater Fuel"]
    }
    # Alfabetik ilk deger yerine en sik deger: arayuz anlamli bir evle aciliyor
    ui_defaults = {
        col: str(X_train[col].mode().iloc[0])
        for col in ["House Type", "Room Count", "Floor", "Heater Type",
                    "Heater Fuel", "Hand"]
    }
    ui_ranges = {
        col: {"min": float(X_train[col].min()), "max": float(X_train[col].max()),
              "median": float(X_train[col].median())}
        for col in ["House Age", "House Size", "Bathroom Count"]
    }

    # Konum agacini egitim verisinden uret. Repodaki data.json isimleri
    # BUYUK HARF ("ADANA", "AKPINAR MAH"), egitim verisi ise "Adana",
    # "Akpinar" -> app.py'nin gonderdigi her konum model icin gorulmemis
    # kategoriydi ve en guclu ozellik (Town) bosa gidiyordu.
    loc = (X_train[["City", "Town", "Neighborhood"]].astype(str)
           .drop_duplicates().sort_values(["City", "Town", "Neighborhood"]))
    locations = {
        city: {town: sorted(g2["Neighborhood"].unique().tolist())
               for town, g2 in g1.groupby("Town")}
        for city, g1 in loc.groupby("City")
    }
    Path(args.locations).write_text(
        json.dumps(locations, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nKonum -> %s  (%d il, %d ilce, %d mahalle)"
          % (args.locations, len(locations),
             sum(len(v) for v in locations.values()),
             int(loc["Neighborhood"].nunique())))

    joblib.dump(model, args.out)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "catboost": __import__("catboost").__version__,
        "sklearn": __import__("sklearn").__version__,
        "pandas": pd.__version__,
        "rows_raw": int(len(raw)),
        "rows_after_cleaning": int(len(df)),
        "rows_train": int(len(y_train)),
        "rows_test": int(len(y_test)),
        "price_bounds_train": {"low": lo, "high": hi},
        "raw_columns": RAW_COLUMNS,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "cv": cv_summary,
        "metrics": results,
        "ui_choices": ui_choices,
        "ui_defaults": ui_defaults,
        "seen_neighborhoods": sorted(X_train["Neighborhood"].dropna().astype(str).unique().tolist()),
        "ui_ranges": ui_ranges,
        "feature_importance": {k: round(float(v), 3) for k, v in imp.items()},
    }
    Path(args.metrics).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nModel  -> %s" % args.out)
    print("Metrik -> %s" % args.metrics)


if __name__ == "__main__":
    main()
