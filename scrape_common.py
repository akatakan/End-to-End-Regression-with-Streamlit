"""Kazima scriptlerinin ortak altyapisi: HTTP, sayi ayristirma, checkpoint.

Eski scriptlerdeki dort yapisal sorun burada cozuluyor:
  * `requests.get`'te timeout yoktu -> tek bir asili baglanti isi durduruyordu
  * sadece 429 ele aliniyordu; 5xx ve baglanti hatalari sessizce kayboluyordu
  * cikti sadece en sonda yaziliyordu -> saatler suren kazimada tek cokme
    her seyi goturuyordu
  * "1.100 m2" gibi Turkce sayilar ham birakiliyordu -> float("1.100") == 1.1
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# Sayfada gorulebilecek para birimleri. Eski scraper `split()[0]` ile sadece
# sayiyi aliyordu; K.K.T.C. ilanlari GBP/EUR oldugu icin 650 GBP veri setine
# "650 TL" olarak giriyordu.
_CURRENCY_PATTERNS = [
    ("TRY", r"\bTL\b|₺"),
    ("USD", r"\bUSD\b|\$"),
    ("EUR", r"\bEUR\b|€"),
    ("GBP", r"\bGBP\b|\bSTG\b|£"),
]

_NUMBER_RE = re.compile(r"\d[\d.\s]*(?:,\d+)?")


class ScrapeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def build_session(user_agent: str = DEFAULT_USER_AGENT,
                  proxy: str | None = None) -> requests.Session:
    """Tek bir Session -> baglanti havuzu ve cerezler korunur.

    `proxy` TEK bir URL string'i olmali ("http://host:port"). Eskiden buraya
    bir liste veriliyordu; requests scheme basina tek string bekledigi icin
    o proxy yapilandirmasi hic calismamisti.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Language": "tr-TR,tr;q=0.9",
    })
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    return session


def polite_get(session: requests.Session, url: str, *, timeout: float = 20.0,
               max_retries: int = 4, base_delay: float = 5.0,
               log=print) -> requests.Response | None:
    """Geri cekilmeli GET. Kalici basarisizlikta None doner, ASLA asili kalmaz.

    None donmesi cagiranin ilerlemesi gerektigi anlamina gelir; eski koddaki
    `continue` ile sonsuz donguye girme durumu boylece imkansiz.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            wait = base_delay * 2 ** (attempt - 1) + random.uniform(0, 2)
            log(f"    baglanti hatasi ({exc.__class__.__name__}), "
                f"{wait:.0f} sn sonra tekrar [{attempt}/{max_retries}]")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response

        if response.status_code in (429, 500, 502, 503, 504):
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = float(retry_after)
            else:
                wait = base_delay * 2 ** (attempt - 1) + random.uniform(0, 2)
            log(f"    HTTP {response.status_code}, {wait:.0f} sn bekleniyor "
                f"[{attempt}/{max_retries}]")
            time.sleep(wait)
            continue

        if response.status_code == 403:
            # 2026-09 itibariyla hepsiemlak Cloudflare arkasinda ve duz HTTP
            # istemcilerine 403 donuyor. Tekrar denemek bunu cozmez; kullaniciya
            # durumu soyleyip duruyoruz.
            log("    HTTP 403 - site otomatik istekleri engelliyor. "
                "robots.txt bu yolu yasaklamiyor ama erisim tarayici "
                "dogrulamasina tabi; siteyle iletisime gecmeden devam etme.")
            return None

        # 404, 410, ... tekrar denemenin anlami yok
        log(f"    HTTP {response.status_code} - atlaniyor")
        return None

    log(f"    {max_retries} denemede alinamadi - atlaniyor")
    return None


def robots_check(base_url: str, target_url: str, user_agent: str = DEFAULT_USER_AGENT,
                 log=print) -> tuple[bool, float | None]:
    """robots.txt'i okur: (izinli mi, tanimli crawl-delay).

    Kazimaya baslamadan once sorulur. robots.txt okunamazsa izinli sayilir
    ama bu durum loglanir - sessizce varsaymiyoruz.
    """
    parser = RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        response = requests.get(
            robots_url, timeout=15,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log(f"robots.txt okunamadi ({exc.__class__.__name__}) - izinli varsayiliyor")
        return True, None

    parser.parse(response.text.splitlines())
    allowed = parser.can_fetch(user_agent, target_url)
    delay = parser.crawl_delay(user_agent) or parser.crawl_delay("*")
    return allowed, delay


def sleep_between(low: float, high: float) -> None:
    time.sleep(random.uniform(low, high))


# ---------------------------------------------------------------------------
# Turkce sayi / fiyat ayristirma
# ---------------------------------------------------------------------------

def parse_tr_number(text: str | None) -> float | None:
    """Turkce bicimli sayiyi float'a cevirir.

    tr-TR'de nokta binlik ayiricidir, virgul ondaliktir:
        "1.100"    -> 1100.0     (eski kod bunu 1.1 yapiyordu)
        "1.100,50" -> 1100.5
        "120 m2"   -> 120.0
    """
    if text is None:
        return None
    match = _NUMBER_RE.search(str(text))
    if not match:
        return None
    raw = match.group(0).replace(" ", "").replace(".", "")
    raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_price(text: str | None) -> tuple[float | None, str | None]:
    """Fiyat metnini (tutar, para birimi) olarak dondurur.

    Para birimini dusurmek yerine kayit altina aliyoruz ki TL disi ilanlar
    veri setine TL gibi karismasin.
    """
    if text is None:
        return None, None
    amount = parse_tr_number(text)
    currency = None
    for code, pattern in _CURRENCY_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            currency = code
            break
    return amount, currency


# ---------------------------------------------------------------------------
# Checkpoint / JSONL
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict) -> None:
    """Her kaydi aninda diske yaz. Cokme halinde en fazla 1 kayit kaybedilir."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # yarim yazilmis son satir (cokme aninda) - sessizce atla
                print(f"  uyari: {path.name}:{line_no} bozuk satir, atlandi")


def done_urls(path: Path) -> set[str]:
    """Daha once islenmis URL'ler -> kazima kaldigi yerden devam eder."""
    return {r["url"] for r in read_jsonl(path) if r.get("url")}
