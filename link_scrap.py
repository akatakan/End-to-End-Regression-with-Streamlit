"""Kiralik ilan linklerini toplar.

Duzeltilen hatalar:
  * son checkpoint'ten sonraki sayfalar kayboluyordu (dump sadece
    `page % 10 == 0` icin; 1892 sayfada son 2 sayfa hic yazilmiyordu)
  * timeout ve try/except yoktu
  * her checkpoint yeni bir dosya yaziyordu (linkler-1-10, -1-20, ...)
  * calisma yarida kalirsa bastan basliyordu
  * ayni link birden fazla sayfada gorunurse tekrarlaniyordu

Kullanim:
    python link_scrap.py --pages 1892
    python link_scrap.py --pages 1892 --resume     # kaldigi sayfadan devam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup

from scrape_common import build_session, polite_get, robots_check, sleep_between

BASE_URL = "https://www.hepsiemlak.com"
LIST_URL = BASE_URL + "/kiralik?page={page}"


def parse_links(html: bytes) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for card in soup.find_all("a", class_="card-link"):
        href = card.get("href")
        if href:
            links.append(href if href.startswith("http") else BASE_URL + href)
    return links


def load_state(path: Path) -> tuple[list[str], int]:
    """Kayitli linkler ve en son tamamlanan sayfa."""
    if not path.exists():
        return [], 0
    state = json.loads(path.read_text(encoding="utf-8"))
    return state.get("urls", []), int(state.get("last_page", 0))


def save_state(path: Path, urls: list[str], last_page: int) -> None:
    """Atomik yaz: once .tmp, sonra replace. Yazma sirasinda cokme olursa
    onceki checkpoint bozulmadan kalir."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"last_page": last_page, "urls": urls}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def enforce_robots(target_url: str, args) -> tuple[bool, float | None]:
    """Kazimadan once robots.txt'e sor. Yasakliysa --ignore-robots olmadan durur."""
    allowed, delay = robots_check(BASE_URL, target_url)
    print(f"robots.txt: {'izinli' if allowed else 'YASAK'}"
          + (f", crawl-delay {delay} sn" if delay else ""))
    if not allowed and not args.ignore_robots:
        print("robots.txt bu yolu yasakliyor. Devam etmek icin --ignore-robots "
              "gerekir; once sitenin kullanim kosullarini kontrol et.")
        return False, delay
    return True, delay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=1892, help="son sayfa (dahil)")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--out", default="links.json")
    ap.add_argument("--resume", action="store_true", help="kaldigi sayfadan devam et")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--min-delay", type=float, default=10.0)
    ap.add_argument("--max-delay", type=float, default=20.0)
    ap.add_argument("--proxy", default=None, help="tek proxy URL'i (http://host:port)")
    ap.add_argument("--ignore-robots", action="store_true")
    args = ap.parse_args()

    allowed, crawl_delay = enforce_robots(LIST_URL.format(page=1), args)
    if not allowed:
        return
    if crawl_delay:
        args.min_delay = max(args.min_delay, crawl_delay)
        args.max_delay = max(args.max_delay, crawl_delay * 2)

    out = Path(args.out)
    urls, last_page = load_state(out) if args.resume else ([], 0)
    seen = set(urls)
    start = max(args.start_page, last_page + 1)
    if args.resume and urls:
        print(f"devam: {len(urls)} link kayitli, {start}. sayfadan baslaniyor")

    session = build_session(proxy=args.proxy)
    page = start
    empty_pages = 0

    try:
        while page <= args.pages:
            print(f"{page}. sayfa...")
            response = polite_get(session, LIST_URL.format(page=page), log=print)

            if response is None:
                # Kalici hata: sayfayi atla ama ISI DURDURMA ve checkpoint'i koru.
                print(f"  {page}. sayfa alinamadi, atlaniyor")
                page += 1
                continue

            new = [u for u in parse_links(response.content) if u not in seen]
            found = len(parse_links(response.content))
            seen.update(new)
            urls.extend(new)
            print(f"  {found} link bulundu, {len(new)} yeni (toplam {len(urls)})")

            # Ilanlar bitince site bos sayfa dondurur; ust uste 3 bos sayfa
            # gorursek --pages degerini beklemeden duruyoruz.
            empty_pages = empty_pages + 1 if found == 0 else 0
            if empty_pages >= 3:
                print("  ust uste 3 bos sayfa - kazima bitti")
                break

            if page % args.checkpoint_every == 0:
                save_state(out, urls, page)
                print(f"  checkpoint: {len(urls)} link -> {out}")

            page += 1
            sleep_between(args.min_delay, args.max_delay)
    except KeyboardInterrupt:
        print("\nkullanici durdurdu")
    finally:
        # Dongu nasil biterse bitsin (bitis, hata, Ctrl+C) son durum yazilir.
        # Eski kodda son sayfalar bu yuzden kayboluyordu.
        save_state(out, urls, page - 1)
        print(f"\n{len(urls)} link kaydedildi -> {out} (son sayfa: {page - 1})")


if __name__ == "__main__":
    main()
