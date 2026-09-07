"""Kiralik ev fiyati tahmin arayuzu."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import RAW_COLUMNS

MODEL_PATH = Path("model_pipeline.pkl")
META_PATH = Path("metrics.json")
LOCATION_PATH = Path("locations.json")

st.set_page_config(page_title="Kiralik Ev Fiyati Tahmini", page_icon="🏠", layout="wide")


# Streamlit her widget degisiminde scripti bastan calistirir. Cache olmadan
# 1,2 MB model + 13,6 MB JSON her tiklamada yeniden yukleniyordu.
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_locations() -> dict:
    """Egitim verisindeki il/ilce/mahalle agaci (train.py uretiyor).

    Repodaki data.json'i kullanmiyoruz: oradaki isimler BUYUK HARF ve
    egitim verisiyle eslesmiyor -> model her konumu "gorulmemis" sayiyordu.
    """
    return json.loads(LOCATION_PATH.read_text(encoding="utf-8"))


model = load_model()
meta = load_meta()
locations = load_locations()

choices = meta["ui_choices"]
ranges = meta["ui_ranges"]
defaults = meta["ui_defaults"]


def _idx(col: str) -> int:
    """Varsayilan olarak egitim verisindeki en sik degeri sec."""
    try:
        return choices[col].index(defaults[col])
    except (KeyError, ValueError):
        return 0

# --- konum -----------------------------------------------------------------
city = st.sidebar.selectbox("Şehir", sorted(locations))
county = st.sidebar.selectbox("İlçe", sorted(locations[city]))
neighborhood = st.sidebar.selectbox("Mahalle", locations[city][county])
st.sidebar.caption("Listede sadece eğitim verisinde geçen konumlar var.")

# --- ev ozellikleri --------------------------------------------------------
# Secenekler egitim verisinden geliyor: arayuz ile model bir daha ayrisamaz.
house_type = st.sidebar.selectbox("Yapı Tipi", choices["House Type"], index=_idx("House Type"))
room_count = st.sidebar.selectbox("Oda Sayısı", choices["Room Count"], index=_idx("Room Count"))
floor = st.sidebar.selectbox("Bulunduğu Kat", choices["Floor"], index=_idx("Floor"))
heater_type = st.sidebar.selectbox("Isıtma Tipi", choices["Heater Type"], index=_idx("Heater Type"))
heater_fuel = st.sidebar.selectbox("Isıtma Yakıtı", choices["Heater Fuel"], index=_idx("Heater Fuel"))

age = st.sidebar.number_input(
    "Bina Yaşı", 0, int(ranges["House Age"]["max"]), int(ranges["House Age"]["median"])
)
house_size = st.sidebar.number_input(
    "Evin Alanı (m²)", 1, int(ranges["House Size"]["max"]),
    int(ranges["House Size"]["median"])
)
bath_count = st.sidebar.number_input(
    "Banyo Sayısı", 1, int(ranges["Bathroom Count"]["max"]), 1
)
furniture = st.sidebar.selectbox("Eşyalı mı?", ["Hayır", "Evet"])
hand = st.sidebar.selectbox("Kaçıncı Sahibi", ["İkinci El", "Sıfır"])

# --- ozet ------------------------------------------------------------------
st.title("Kiralık Ev Fiyatı Tahmini")
st.header("Tahmin Edilmesi İstenen Ev Bilgileri", divider="red")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write(f"**İl:** {city}")
    st.write(f"**Bina Yaşı:** {age}")
    st.write(f"**Eşyalı:** {furniture}")
with col2:
    st.write(f"**İlçe:** {county}")
    st.write(f"**Alan:** {house_size} m²")
    st.write(f"**Banyo:** {bath_count}")
with col3:
    st.write(f"**Mahalle:** {neighborhood}")
    st.write(f"**Oda Sayısı:** {room_count}")
    st.write(f"**Sahiplik:** {hand}")
with col4:
    st.write(f"**Yapı Tipi:** {house_type}")
    st.write(f"**Kat:** {floor}")
    st.write(f"**Isıtma:** {heater_type} / {heater_fuel}")

st.header("", divider="red")

row = pd.DataFrame([{
    "City": city,
    "Town": county,
    "Neighborhood": neighborhood,
    "House Type": house_type,
    "House Age": age,
    "House Size": house_size,
    "Room Count": room_count,
    "Floor": floor,
    "Furniture": furniture,
    "Bathroom Count": bath_count,
    "Hand": hand,
    "Heater Type": heater_type,
    "Heater Fuel": heater_fuel,
}])[RAW_COLUMNS]

if st.button("Tahmin Et", type="primary"):
    pred = float(model.predict(row)[0])

    # Nokta tahmini tek basina yaniltici; test setindeki medyan mutlak yuzde
    # hatasini bant olarak gosteriyoruz.
    medape = meta["metrics"]["catboost_in_range"]["MedAPE%"] / 100
    low, high = pred * (1 - medape), pred * (1 + medape)

    st.subheader(f"Tahmini Kira: {pred:,.0f} TL".replace(",", "."))
    st.caption(
        f"Olası aralık: {low:,.0f} – {high:,.0f} TL".replace(",", ".")
        + f"  (test setinde tahminlerin yarısı ±%{medape * 100:.0f} içinde)"
    )

    bounds = meta["price_bounds_train"]
    if not (bounds["low"] <= pred <= bounds["high"]):
        st.warning(
            f"Tahmin, modelin eğitildiği {bounds['low']:,.0f}–{bounds['high']:,.0f} TL "
            "aralığının dışında; bu değere güvenme.".replace(",", ".")
        )

with st.expander("Model hakkında"):
    m = meta["metrics"]["catboost_in_range"]
    st.write(
        f"Eğitim: {meta['rows_train']:,} ilan · Test: {meta['rows_test']:,} ilan · "
        f"{meta['trained_at'][:10]}".replace(",", ".")
    )
    st.write(
        f"Test seti (eğitim aralığı içi): MAPE %{m['MAPE%']:.1f} · "
        f"MedAPE %{m['MedAPE%']:.1f} · R² {m['R2']:.3f} · "
        f"tahminlerin %{m['within20%']:.0f}'i gerçek değerin ±%20 bandında"
    )
    st.write("Veri kaynağı: hepsiemlak kiralık ilanları (tek seferlik kazıma).")
