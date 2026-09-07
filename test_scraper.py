"""Kazima mantiginin testleri - canli siteye istek atmadan.

HTML fixture'lari, eski scriptlerin kullandigi seciciler baz alinarak
yazildi (`ul.short-info-list`, `p.fz24-text.price`, `span.txt` + kardes span,
`p.stale-warning__text`, `a.card-link`). Sayfa yapisi degisirse once burasi
guncellenmeli.

    python test_scraper.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

import info_scrapy

from info_scrapy import load_links, parse_listing, to_row, write_csv
from link_scrap import load_state, parse_links, save_state
from scrape_common import (
    append_jsonl,
    done_urls,
    parse_price,
    parse_tr_number,
    polite_get,
    read_jsonl,
)


def listing_html(price="16.000 TL", size="125 m2", details=None, stale=False) -> bytes:
    details = details if details is not None else {
        "Bulunduğu Kat": "Ara Kat",
        "Bina Yaşı": "3",
        "Isınma Tipi": "Kombi",
        "Eşya Durumu": "Eşyasız",
        "Banyo Sayısı": "2",
        "Yapının Durumu": "İkinci El",
        "Yakıt Tipi": "Doğalgaz",
        "Konut Tipi": "Ara Kat Dubleks",
    }
    rows = "".join(
        f'<div><span class="txt">{k}</span><span>{v}</span></div>'
        for k, v in details.items()
    )
    warning = '<p class="stale-warning__text">Yayindan kaldirildi</p>' if stale else ""
    return f"""
    <html><body>
      {warning}
      <ul class="short-info-list">
        <li>Ankara</li><li>Mamak</li><li>Akşemsettin</li>
        <li>Kiralık</li><li>Daire</li><li>3 + 1</li><li>{size}</li>
      </ul>
      <p class="fz24-text price">{price}</p>
      {rows}
    </body></html>
    """.encode("utf-8")


class TestNumberParsing(unittest.TestCase):
    def test_thousand_separator(self):
        # Eski kodun bozuldugu yer: float("1.100") == 1.1
        self.assertEqual(parse_tr_number("1.100"), 1100.0)
        self.assertEqual(parse_tr_number("1.100 m2"), 1100.0)
        self.assertEqual(parse_tr_number("2.195 m²"), 2195.0)

    def test_plain_and_decimal(self):
        self.assertEqual(parse_tr_number("125"), 125.0)
        self.assertEqual(parse_tr_number("125 m2"), 125.0)
        self.assertEqual(parse_tr_number("1.100,50"), 1100.5)
        self.assertEqual(parse_tr_number("12,5"), 12.5)

    def test_missing(self):
        self.assertIsNone(parse_tr_number(None))
        self.assertIsNone(parse_tr_number("Belirtilmemiş"))


class TestPriceParsing(unittest.TestCase):
    def test_try(self):
        self.assertEqual(parse_price("16.000 TL"), (16000.0, "TRY"))
        self.assertEqual(parse_price("1.250.000 ₺"), (1250000.0, "TRY"))

    def test_foreign_currency_is_kept(self):
        # K.K.T.C. ilanlari: eski kod bunlari "650 TL" olarak kaydediyordu
        self.assertEqual(parse_price("650 GBP"), (650.0, "GBP"))
        self.assertEqual(parse_price("1.200 €"), (1200.0, "EUR"))
        self.assertEqual(parse_price("$900"), (900.0, "USD"))

    def test_unknown_currency(self):
        amount, currency = parse_price("650")
        self.assertEqual(amount, 650.0)
        self.assertIsNone(currency)


class TestListingParsing(unittest.TestCase):
    def test_full_listing(self):
        row = to_row(parse_listing(listing_html(), "https://x/ilan/1"))
        self.assertEqual(row["City"], "Ankara")
        self.assertEqual(row["Town"], "Mamak")
        self.assertEqual(row["Neighborhood"], "Akşemsettin")
        self.assertEqual(row["House Type"], "Daire")
        self.assertEqual(row["Room Count"], "3 + 1")
        self.assertEqual(row["House Size"], 125.0)
        self.assertEqual(row["Price"], 16000.0)
        self.assertEqual(row["Currency"], "TRY")
        self.assertEqual(row["Floor"], "Ara Kat")
        self.assertEqual(row["Bathroom Count"], 2.0)
        self.assertIs(row["Furniture"], False)
        self.assertEqual(row["Hand"], "İkinci El")
        self.assertEqual(row["Apartment Type"], "Ara Kat Dubleks")

    def test_big_house_size_not_mangled(self):
        row = to_row(parse_listing(listing_html(size="1.100 m2"), "u"))
        self.assertEqual(row["House Size"], 1100.0)

    def test_building_structure_does_not_leak_into_hand(self):
        # "Yapinin Durumu" bazi ilanlarda yapi turunu veriyor
        html = listing_html(details={"Yapının Durumu": "Betonarme"})
        row = to_row(parse_listing(html, "u"))
        self.assertIsNone(row["Hand"])
        self.assertEqual(row["Building Structure"], "Betonarme")

    def test_zero_age(self):
        row = to_row(parse_listing(listing_html(details={"Bina Yaşı": "Sıfır"}), "u"))
        self.assertEqual(row["House Age"], 0.0)

    def test_missing_fields_are_none_not_crash(self):
        row = to_row(parse_listing(listing_html(details={}), "u"))
        self.assertIsNone(row["Floor"])
        self.assertIsNone(row["Heater Type"])
        self.assertIsNone(row["Furniture"])
        self.assertEqual(row["City"], "Ankara")

    def test_stale_listing(self):
        self.assertIsNone(parse_listing(listing_html(stale=True), "u"))

    def test_missing_short_info(self):
        self.assertIsNone(parse_listing(b"<html><body></body></html>", "u"))


class TestLinkParsing(unittest.TestCase):
    def test_relative_and_absolute(self):
        html = (b'<a class="card-link" href="/ilan/1">a</a>'
                b'<a class="card-link" href="https://www.hepsiemlak.com/ilan/2">b</a>'
                b'<a class="other" href="/ilan/3">c</a>')
        self.assertEqual(parse_links(html), [
            "https://www.hepsiemlak.com/ilan/1",
            "https://www.hepsiemlak.com/ilan/2",
        ])

    def test_empty_page(self):
        self.assertEqual(parse_links(b"<html></html>"), [])


class TestCheckpointing(unittest.TestCase):
    def test_link_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "links.json"
            save_state(path, ["a", "b"], 20)
            urls, last = load_state(path)
            self.assertEqual((urls, last), (["a", "b"], 20))
            self.assertEqual(load_links(path), ["a", "b"])

    def test_load_state_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_state(Path(tmp) / "yok.json"), ([], 0))

    def test_resume_skips_done_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "listings.jsonl"
            append_jsonl(path, {"url": "u1", "price": 1})
            append_jsonl(path, {"url": "u2", "stale": True})
            self.assertEqual(done_urls(path), {"u1", "u2"})

    def test_truncated_last_line_is_skipped(self):
        # Cokme aninda yarim kalan satir tum dosyayi okunamaz yapmamali
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "listings.jsonl"
            append_jsonl(path, {"url": "u1"})
            with path.open("a", encoding="utf-8") as fh:
                fh.write('{"url": "u2", "pri')
            self.assertEqual(done_urls(path), {"u1"})


class TestCsvOutput(unittest.TestCase):
    def test_csv_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl, csv = Path(tmp) / "l.jsonl", Path(tmp) / "e.csv"
            append_jsonl(jsonl, parse_listing(listing_html(), "u1"))
            append_jsonl(jsonl, parse_listing(listing_html(size="1.100 m2"), "u2"))
            append_jsonl(jsonl, {"url": "u3", "stale": True})   # fiyatsiz -> dusmeli
            write_csv(jsonl, csv)
            import pandas as pd
            df = pd.read_csv(csv)
            self.assertEqual(len(df), 2)
            self.assertEqual(sorted(df["House Size"]), [125.0, 1100.0])
            self.assertIn("Currency", df.columns)


class FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class TestRobotsGate(unittest.TestCase):
    def test_disallowed_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            links, jsonl = tmp / "l.json", tmp / "s.jsonl"
            links.write_text(json.dumps(["u1"]), encoding="utf-8")
            argv = ["info_scrapy.py", "--links", str(links), "--jsonl", str(jsonl),
                    "--csv", str(tmp / "e.csv")]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("info_scrapy.robots_check", return_value=(False, None)),
                mock.patch("info_scrapy.polite_get") as get,
                mock.patch("info_scrapy.build_session"),
                mock.patch("builtins.print"),
            ):
                info_scrapy.main()
            get.assert_not_called()


class TestRetryBehaviour(unittest.TestCase):
    """polite_get her yolda SONLANMALI - eski koddaki `continue` sonsuzdu."""

    def _session(self, responses):
        calls = []

        class S:
            def get(self, url, timeout=None):
                calls.append(url)
                item = responses[min(len(calls) - 1, len(responses) - 1)]
                if isinstance(item, Exception):
                    raise item
                return item

        return S(), calls

    def test_persistent_429_gives_up(self):
        session, calls = self._session([FakeResponse(429)])
        with mock.patch("scrape_common.time.sleep"):
            result = polite_get(session, "u", max_retries=4, log=lambda *a: None)
        self.assertIsNone(result)
        self.assertEqual(len(calls), 4)

    def test_404_is_not_retried(self):
        session, calls = self._session([FakeResponse(404)])
        with mock.patch("scrape_common.time.sleep"):
            self.assertIsNone(polite_get(session, "u", log=lambda *a: None))
        self.assertEqual(len(calls), 1)

    def test_recovers_after_transient_error(self):
        ok = FakeResponse(200, b"icerik")
        session, calls = self._session([requests.ConnectionError("bum"), ok])
        with mock.patch("scrape_common.time.sleep"):
            result = polite_get(session, "u", log=lambda *a: None)
        self.assertIsNotNone(result)
        self.assertEqual(result.content, b"icerik")
        self.assertEqual(len(calls), 2)

    def test_retry_after_header_is_respected(self):
        session, _ = self._session([FakeResponse(429, headers={"Retry-After": "7"})])
        with mock.patch("scrape_common.time.sleep") as slept:
            polite_get(session, "u", max_retries=2, log=lambda *a: None)
        self.assertEqual([c.args[0] for c in slept.call_args_list], [7.0, 7.0])


class TestScrapeLoopAlwaysAdvances(unittest.TestCase):
    """Kalici hata veren linkler donguyu kilitlememeli (eski kodun bugu)."""

    def _run(self, get_result):
        """Kaziyiciyi sahte HTTP ile calistirir.

        Sonuclari gecici dizin silinmeden ONCE okuyup donduruyoruz.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            links, jsonl, csv = tmp / "l.json", tmp / "s.jsonl", tmp / "e.csv"
            links.write_text(json.dumps(["u1", "u2", "u3"]), encoding="utf-8")
            argv = ["info_scrapy.py", "--links", str(links), "--jsonl", str(jsonl),
                    "--csv", str(csv), "--min-delay", "0", "--max-delay", "0"]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("info_scrapy.polite_get", side_effect=get_result) as g,
                # robots kontrolu ag istegi yapiyor - testler agdan bagimsiz olmali
                mock.patch("info_scrapy.robots_check", return_value=(True, None)),
                mock.patch("info_scrapy.build_session"),
                mock.patch("builtins.print"),
            ):
                info_scrapy.main()
            return {
                "calls": g.call_count,
                "written": jsonl.exists(),
                "done": done_urls(jsonl),
                "rows": [to_row(r) for r in read_jsonl(jsonl)],
            }

    def test_all_links_fail_loop_still_ends(self):
        out = self._run(lambda *a, **k: None)
        self.assertEqual(out["calls"], 3)        # her link tam bir kez denendi
        self.assertFalse(out["written"])         # kayit yok ama dongu kilitlenmedi

    def test_stale_listings_are_marked_so_resume_skips_them(self):
        out = self._run(lambda *a, **k: FakeResponse(200, listing_html(stale=True)))
        self.assertEqual(out["calls"], 3)
        self.assertEqual(out["done"], {"u1", "u2", "u3"})

    def test_successful_scrape_is_written_per_record(self):
        out = self._run(lambda *a, **k: FakeResponse(200, listing_html()))
        self.assertEqual(out["done"], {"u1", "u2", "u3"})
        self.assertTrue(all(r["Price"] == 16000.0 for r in out["rows"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
