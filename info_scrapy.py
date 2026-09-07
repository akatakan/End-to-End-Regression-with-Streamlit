"""Ilan detaylarini kazir ve emlak.csv uretir.

Duzeltilen hatalar:
  * `while step < len(house_links)-1` -> son link hep atlaniyordu
  * `except` blogunda `step` artmiyordu -> kalici hata veren tek bir linkte
    SONSUZ DONGU
  * CSV sadece en sonda yaziliyordu -> saatler suren kazimada tek cokme
    her seyi goturuyordu
  * `proxies = {"http": <liste>}` -> requests scheme basina tek string bekler;
    o proxy yapilandirmasi hic calismamisti (ve proxy-list.txt repoda yoktu)
  * fiyattaki para birimi dusuruluyordu -> GBP/EUR ilanlar TL gibi kaydediliyordu
  * "1.100 m2" ham birakiliyordu -> pandas onu 1.1 olarak okuyordu
  * "Yapinin Durumu" bazi ilanlarda yapi turunu ("Betonarme") veriyor, o deger
    Sifir/Ikinci El sutununa karisiyordu
  * `Apartment Type` toplaniyor ama CSV'ye yazilmiyordu

Ham kayitlar JSONL'e yazilir (`listings.jsonl`), CSV oradan uretilir. Boylece
kazimayi tekrarlamadan alan cikarimini degistirebilirsin.

Kullanim:
    python info_scrapy.py --links links.json          # kazi (kaldigi yerden)
    python info_scrapy.py --csv-only                  # JSONL -> emlak.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from scrape_common import (
    append_jsonl,
    build_session,
    done_urls,
    parse_price,
    parse_tr_number,
    polite_get,
    read_jsonl,
    robots_check,
    sleep_between,
)

# "Yapinin Durumu" alani iki farkli sey dondurebiliyor. Sadece bunlar sahiplik
# bilgisidir; digerleri (Betonarme, Celik, Ahsap...) yapi turudur ve ayri
# sutuna yaziliyor.
HAND_VALUES = {"Sıfır", "İkinci El", "Yapım Aşamasında"}

DETAIL_LABELS = {
    "Bulunduğu Kat": "floor",
    "Bina Yaşı": "house_age",
    "Isınma Tipi": "heater_type",
    "Eşya Durumu": "furniture",
    "Banyo Sayısı": "bath_count",
    "Yapının Durumu": "building_state",
    "Yakıt Tipi": "heater_fuel",
    "Konut Tipi": "apartment_type",
}

CSV_COLUMNS = [
    "City", "Town", "Neighborhood", "Apartment Type", "House Type", "House Age",
    "House Size", "Room Count", "Floor", "Furniture", "Bathroom Count", "Hand",
    "Building Structure", "Heater Type", "Heater Fuel", "Currency", "Price", "url",
]


def parse_listing(html: bytes, url: str) -> dict | None:
    """Ilan sayfasindan ham alanlari cikarir. Eksik alan hata degildir."""
    soup = BeautifulSoup(html, "lxml")

    if soup.find("p", class_="stale-warning__text"):
        return None  # yayindan kaldirilmis ilan

    ul = soup.find("ul", class_="short-info-list")
    if not ul:
        return None

    short_info = [li.get_text(strip=True) for li in ul.find_all("li")]

    price_tag = soup.find("p", class_="fz24-text price")
    amount, currency = parse_price(price_tag.get_text(strip=True) if price_tag else None)

    # Etiket -> deger. Konuma degil etikete bakiyoruz; sayfa duzeni degisirse
    # yanlis sutuna veri yazmak yerine alan bos kalir.
    details: dict[str, str] = {}
    for label_tag in soup.find_all("span", class_="txt"):
        key = DETAIL_LABELS.get(label_tag.get_text(strip=True))
        if not key:
            continue
        value_tag = label_tag.find_next_sibling("span")
        if value_tag is not None:
            details[key] = value_tag.get_text(strip=True)

    return {
        "url": url,
        "price": amount,
        "currency": currency,
        # Ham liste de saklaniyor: alan cikarimi degisirse yeniden kazimaya
        # gerek kalmadan JSONL'den tureyebilir.
        "short_info": short_info,
        "details": details,
    }


def to_row(record: dict) -> dict:
    """Ham JSONL kaydi -> CSV satiri."""
    short = record.get("short_info") or []
    d = record.get("details") or {}

    def at(i: int) -> str | None:
        return short[i] if len(short) > i else None

    building_state = d.get("building_state")
    hand = building_state if building_state in HAND_VALUES else None
    structure = None if building_state in HAND_VALUES else building_state

    furniture = d.get("furniture")
    if furniture == "Eşyalı":
        furniture_flag: bool | None = True
    elif furniture == "Eşyasız":
        furniture_flag = False
    else:
        furniture_flag = None

    age_raw = d.get("house_age")
    house_age = 0.0 if age_raw == "Sıfır" else parse_tr_number(age_raw)

    return {
        "City": at(0),
        "Town": at(1),
        "Neighborhood": at(2),
        "Apartment Type": d.get("apartment_type"),
        "House Type": at(4),
        "House Age": house_age,
        # parse_tr_number: "1.100 m2" -> 1100.0 (eskiden 1.1 oluyordu)
        "House Size": parse_tr_number(at(6)),
        "Room Count": at(5),
        "Floor": d.get("floor"),
        "Furniture": furniture_flag,
        "Bathroom Count": parse_tr_number(d.get("bath_count")),
        "Hand": hand,
        "Building Structure": structure,
        "Heater Type": d.get("heater_type"),
        "Heater Fuel": d.get("heater_fuel"),
        "Currency": record.get("currency"),
        "Price": record.get("price"),
        "url": record.get("url"),
    }


def write_csv(jsonl: Path, csv_path: Path) -> None:
    rows = [to_row(r) for r in read_jsonl(jsonl)]
    if not rows:
        print("JSONL bos, CSV yazilmadi")
        return
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    before = len(df)
    df = df.drop_duplicates(subset="url").dropna(subset=["Price"])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"{csv_path}: {len(df)} satir ({before - len(df)} tekrar/fiyatsiz atildi)")
    if "Currency" in df:
        counts = df["Currency"].value_counts(dropna=False).to_dict()
        print(f"  para birimi dagilimi: {counts}")


def load_links(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):          # link_scrap.py'nin state dosyasi
        return list(data.get("urls", []))
    return list(data)                   # duz liste de kabul


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", default="links.json")
    ap.add_argument("--jsonl", default="listings.jsonl")
    ap.add_argument("--csv", default="emlak.csv")
    ap.add_argument("--csv-only", action="store_true",
                    help="kazima yapma, mevcut JSONL'den CSV uret")
    ap.add_argument("--limit", type=int, default=None, help="en fazla N ilan kaz")
    ap.add_argument("--min-delay", type=float, default=1.0)
    ap.add_argument("--max-delay", type=float, default=5.0)
    ap.add_argument("--proxy", default=None, help="tek proxy URL'i (http://host:port)")
    ap.add_argument("--ignore-robots", action="store_true")
    args = ap.parse_args()

    jsonl = Path(args.jsonl)

    if not args.csv_only:
        links = load_links(Path(args.links))
        already = done_urls(jsonl)
        todo = [u for u in links if u not in already]
        if args.limit:
            todo = todo[:args.limit]
        print(f"{len(links)} link | {len(already)} zaten kazinmis | {len(todo)} kalan")

        if todo:
            allowed, delay = robots_check("https://www.hepsiemlak.com", todo[0])
            print(f"robots.txt: {'izinli' if allowed else 'YASAK'}"
                  + (f", crawl-delay {delay} sn" if delay else ""))
            if not allowed and not args.ignore_robots:
                print("robots.txt bu yolu yasakliyor - duruluyor "
                      "(--ignore-robots ile gecebilirsin).")
                return
            if delay:
                args.min_delay = max(args.min_delay, delay)
                args.max_delay = max(args.max_delay, delay * 2)

        session = build_session(proxy=args.proxy)
        stale = failed = 0
        try:
            for i, url in enumerate(todo, 1):
                print(f"[{i}/{len(todo)}] {url}")
                response = polite_get(session, url, log=print)

                # Her yol dongunun ILERLEMESIYLE bitiyor. Eski kodda hata
                # durumunda `step` artmadigi icin ayni link sonsuz deneniyordu.
                if response is None:
                    failed += 1
                else:
                    record = parse_listing(response.content, url)
                    if record is None:
                        stale += 1
                        # Yayindan kalkmis ilani da isaretle ki --resume'da
                        # tekrar denenmesin.
                        append_jsonl(jsonl, {"url": url, "stale": True})
                    else:
                        append_jsonl(jsonl, record)

                sleep_between(args.min_delay, args.max_delay)
        except KeyboardInterrupt:
            print("\nkullanici durdurdu - kayitlar diskte, --resume ile devam eder")

        print(f"\nbitti: {stale} yayindan kalkmis, {failed} alinamadi")

    write_csv(jsonl, Path(args.csv))


if __name__ == "__main__":
    main()
