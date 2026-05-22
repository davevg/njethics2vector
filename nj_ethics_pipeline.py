#!/usr/bin/env python3
"""
End-to-end NJ ethics document pipeline: crawl, discover, download, OCR, embed, and ingest.

This script can run the full sync flow or individual stages:
- Crawl seed NJ legal/ethics pages and discover PDF URLs (with filters/exclusions)
- Download newly discovered PDFs with retry/rate-limit handling
- OCR PDFs via OpenRouter, chunk text, generate embeddings via Ollama,
  and ingest documents/chunks into Postgres + pgvector

Useful modes:
- --dry-run        Discover only (no download/ingest)
- --download-only  Discover + download only
- --ingest-only    Ingest existing local PDFs only
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import random
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urldefrag, urljoin, urlparse

import fitz  # PyMuPDF
import psycopg2
import psycopg2.extras
import requests
from bs4 import BeautifulSoup
from pgvector.psycopg2 import register_vector
from tqdm import tqdm

fitz.TOOLS.mupdf_display_errors(False)

DEFAULT_BASE_URLS = [
    "https://www.nj.gov/education/legal/ethics/index.shtml",
    "https://www.nj.gov/education/legal/commissioner/index.shtml",
]
DEFAULT_EXCLUDED_PREFIXES = [
    "https://www.nj.gov/education/legal/examiners/",
    "https://www.nj.gov/education/legal/sboe/",
    "https://www.nj.gov/education/ethics/docs/",
    "https://www.nj.gov/dca/",
    "https://www.nj.gov/oal/docs/",
]
RETRIABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# ---------- ingest config ----------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OCR_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-vl-32b-instruct")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "1500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
OCR_TIMEOUT_S = int(os.environ.get("OCR_TIMEOUT_S", "300"))
EMBED_TIMEOUT_S = int(os.environ.get("EMBED_TIMEOUT_S", "120"))
TEXT_OUT_DIR = os.environ.get("TEXT_OUT_DIR", "ocr_texts")
OCR_WORKERS = int(os.environ.get("OCR_WORKERS", "5"))
OCR_RETRIES = int(os.environ.get("OCR_RETRIES", "3"))

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "dbname": os.environ.get("DB_NAME", "ethics"),
    "user": os.environ.get("DB_USER", os.environ.get("USER", "")),
    "password": os.environ.get("DB_PASSWORD", "password"),
}


class PoliteHttpClient:
    def __init__(self, request_delay: float, request_jitter: float, timeout: int, max_retries: int, user_agent: str, respect_retry_after: bool = True) -> None:
        self.request_delay = max(0.0, request_delay)
        self.request_jitter = max(0.0, request_jitter)
        self.timeout = timeout
        self.max_retries = max_retries
        self.respect_retry_after = respect_retry_after
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request_ts = 0.0

    def _sleep_for_pacing(self) -> None:
        gap = time.time() - self._last_request_ts
        delay = self.request_delay + (random.uniform(0, self.request_jitter) if self.request_jitter > 0 else 0.0)
        if delay - gap > 0:
            time.sleep(delay - gap)

    def _retry_after_seconds(self, resp: requests.Response) -> float | None:
        if not self.respect_retry_after:
            return None
        ra = resp.headers.get("Retry-After")
        if not ra:
            return None
        try:
            return max(0.0, float(ra))
        except ValueError:
            return None

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._sleep_for_pacing()
            self._last_request_ts = time.time()
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code in RETRIABLE_STATUS_CODES:
                    if attempt >= self.max_retries:
                        resp.raise_for_status()
                    wait = self._retry_after_seconds(resp)
                    time.sleep(wait if wait is not None else min(2**attempt, 12))
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 12))
        if last_error:
            raise last_error
        raise RuntimeError(f"Unknown HTTP failure for {method} {url}")


def normalize_url(base_url: str, href: str) -> str:
    candidate = urljoin(base_url, href.strip())
    cleaned, _ = urldefrag(candidate)
    return cleaned


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def same_domain(url: str, base_domain: str) -> bool:
    return urlparse(url).netloc.lower() == base_domain.lower()


def extract_primary_content(soup: BeautifulSoup):
    for selector in ("main", "article", "#content", ".content", "#main-content", ".main-content"):
        node = soup.select_one(selector)
        if node is not None:
            return node
    return soup.body if soup.body else soup


def should_visit_url(url: str, root_domain: str, same_domain_only: bool, include_regex: re.Pattern[str] | None, exclude_regex: re.Pattern[str] | None) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if same_domain_only and not same_domain(url, root_domain):
        return False
    if include_regex and not include_regex.search(url):
        return False
    if exclude_regex and exclude_regex.search(url):
        return False
    return True


def is_excluded_by_prefix(url: str, excluded_prefixes: list[str]) -> bool:
    u = url.strip().lower()
    for p in excluded_prefixes:
        if u.startswith(p.strip().lower()):
            return True
    return False


def crawl_for_pdfs(
    client: PoliteHttpClient,
    base_urls: list[str],
    max_depth: int,
    same_domain_only: bool,
    include_pattern: str | None,
    exclude_pattern: str | None,
    excluded_prefixes: list[str],
    max_pages: int,
    progress_every: int,
) -> tuple[list[str], dict[str, str]]:
    include_regex = re.compile(include_pattern) if include_pattern else None
    exclude_regex = re.compile(exclude_pattern) if exclude_pattern else None
    visited_pages: set[str] = set()
    discovered_pdfs: set[str] = set()
    discovered_from: dict[str, str] = {}
    enqueued_pages: set[str] = set(base_urls)
    queue: deque[tuple[str, int, str]] = deque((base, 0, urlparse(base).netloc) for base in base_urls)
    pages_processed = 0

    while queue:
        if max_pages > 0 and pages_processed >= max_pages:
            print(f"[crawl] reached max-pages={max_pages}; stopping crawl")
            break

        page_url, depth, root_domain = queue.popleft()
        if page_url in visited_pages:
            continue
        if not should_visit_url(page_url, root_domain, same_domain_only, include_regex, exclude_regex):
            continue
        if is_excluded_by_prefix(page_url, excluded_prefixes):
            continue
        try:
            resp = client.request("GET", page_url)
        except requests.RequestException as exc:
            print(f"[crawl] skip {page_url} (error: {exc})")
            continue
        if "text/html" not in (resp.headers.get("Content-Type") or "").lower():
            visited_pages.add(page_url)
            pages_processed += 1
            continue
        visited_pages.add(page_url)
        pages_processed += 1
        if progress_every > 0 and pages_processed % progress_every == 0:
            print(
                f"[crawl] progress pages={pages_processed} queued={len(queue)} "
                f"pdfs={len(discovered_pdfs)}"
            )
        section = extract_primary_content(BeautifulSoup(resp.text, "html.parser"))
        for a in section.find_all("a", href=True):
            target = normalize_url(page_url, a["href"])
            if is_pdf_url(target):
                if (
                    should_visit_url(target, root_domain, same_domain_only, include_regex, exclude_regex)
                    and not is_excluded_by_prefix(target, excluded_prefixes)
                    and target not in discovered_pdfs
                ):
                    discovered_pdfs.add(target)
                    discovered_from[target] = page_url
                continue
            if (
                depth < max_depth
                and should_visit_url(target, root_domain, same_domain_only, include_regex, exclude_regex)
                and not is_excluded_by_prefix(target, excluded_prefixes)
                and target not in visited_pages
                and target not in enqueued_pages
            ):
                queue.append((target, depth + 1, root_domain))
                enqueued_pages.add(target)
    return sorted(discovered_pdfs), discovered_from


def safe_filename_from_url(pdf_url: str) -> str:
    name = Path(urlparse(pdf_url).path).name
    return name if name else f"doc_{abs(hash(pdf_url))}.pdf"


def download_pdfs(client: PoliteHttpClient, pdf_urls: Iterable[str], pdf_dir: Path) -> tuple[int, int, int]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failed = 0
    for pdf_url in pdf_urls:
        out_path = pdf_dir / safe_filename_from_url(pdf_url)
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        try:
            resp = client.request("GET", pdf_url, stream=True)
            with out_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
            downloaded += 1
        except requests.RequestException as exc:
            failed += 1
            out_path.unlink(missing_ok=True)
            print(f"[download] FAIL: {pdf_url} ({exc})")
    return downloaded, skipped, failed


def write_manifest(manifest_path: Path, pdf_urls: list[str], source_map: dict[str, str]) -> None:
    payload = {"generated_at": int(time.time()), "count": len(pdf_urls), "items": [{"pdf_url": url, "discovered_from": source_map.get(url, "")} for url in pdf_urls]}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def discovered_output_rows(
    pdf_urls: list[str],
    pdf_dir: Path,
    print_new_only: bool,
    new_check_scope: str,
) -> list[dict[str, str | bool]]:
    local_names = {p.name for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"} if pdf_dir.exists() else set()
    ingested_names: set[str] = set()
    if new_check_scope in ("ingested", "both"):
        try:
            conn = connect()
            try:
                ingested_names = ingested_filenames(conn)
            finally:
                conn.close()
        except Exception as e:
            print(f"[discover] warning: could not read ingested filenames ({e})")

    rows: list[dict[str, str | bool]] = []
    for url in pdf_urls:
        filename = safe_filename_from_url(url)
        is_downloaded = filename in local_names
        is_ingested = filename in ingested_names
        if new_check_scope == "downloaded":
            is_new = not is_downloaded
        elif new_check_scope == "ingested":
            is_new = not is_ingested
        else:
            is_new = (not is_downloaded) and (not is_ingested)

        if print_new_only and not is_new:
            continue

        rows.append(
            {
                "pdf_url": url,
                "filename": filename,
                "is_new": is_new,
                "downloaded": is_downloaded,
                "ingested": is_ingested,
            }
        )
    return rows


def write_discovered_output(path: Path, rows: list[dict[str, str | bool]]) -> None:
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['pdf_url']}\n")


# ---------- ingest internals ----------
def openrouter_ocr_page(image_png_bytes: bytes) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    b64 = base64.b64encode(image_png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    r = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": OCR_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Extract all text from this document image exactly as it appears. Preserve paragraphs and line breaks. Do not summarize, comment, or add headings."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        },
        timeout=OCR_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def ocr_page_with_retry(image_png_bytes: bytes, retries: int = OCR_RETRIES) -> str:
    for attempt in range(retries + 1):
        try:
            return openrouter_ocr_page(image_png_bytes)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if attempt >= retries or code not in RETRIABLE_STATUS_CODES:
                raise
            time.sleep(min(2**attempt, 8))
        except requests.RequestException:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("OCR retry loop exited unexpectedly")


def ollama_embed(text: str) -> List[float]:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=EMBED_TIMEOUT_S)
    r.raise_for_status()
    vec = r.json().get("embedding")
    if not vec:
        raise RuntimeError("Empty embedding")
    if len(vec) != EMBED_DIM:
        raise RuntimeError(f"Embedding dim mismatch: got {len(vec)}, expected {EMBED_DIM}")
    return vec


def pdf_pages_to_text(pdf_path: Path, ocr_workers: int) -> tuple[str, int]:
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        page_images = [page.get_pixmap(dpi=RENDER_DPI).tobytes("png") for page in doc]
    if page_count == 0:
        return "", 0
    workers = max(1, min(ocr_workers, page_count))
    results = [""] * page_count
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(ocr_page_with_retry, img): idx for idx, img in enumerate(page_images)}
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"[OCR_ERROR page {idx + 1}: {e}]"
    return "\n\n".join(results).strip(), page_count


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def normalize_text(s: str) -> str:
    return _NL_RE.sub("\n\n", _WS_RE.sub(" ", s.replace("\r\n", "\n").replace("\r", "\n"))).strip()


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_text(text)
    if len(text) <= size:
        return [text] if text else []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            ws = max(start + int(size * 0.75), start + 1)
            best = -1
            for sep in ("\n\n", "\n", ". ", "? ", "! "):
                idx = text.rfind(sep, ws, end)
                if idx > best:
                    best = idx + len(sep)
            if best > 0:
                end = best
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def connect():
    conn = psycopg2.connect(**{k: v for k, v in DB_CONFIG.items() if v != ""})
    register_vector(conn)
    return conn


def ingested_filenames(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM ethics_documents")
        return {row[0] for row in cur.fetchall()}


def delete_existing(conn, filename: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ethics_documents WHERE filename = %s", (filename,))


def insert_document_and_chunks(conn, filename: str, full_text: str, page_count: int, chunks: List[str], embeddings: List[List[float]]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ethics_documents (filename, page_count, char_count, ocr_model, full_text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (filename, page_count, len(full_text), OCR_MODEL, full_text),
        )
        doc_id = cur.fetchone()[0]
        rows = [(doc_id, i, c, len(c.split()), e) for i, (c, e) in enumerate(zip(chunks, embeddings))]
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO ethics_chunks (document_id, chunk_index, content, token_count, embedding) VALUES %s",
            rows,
            template="(%s, %s, %s, %s, %s)",
            page_size=100,
        )
    conn.commit()


def write_text_file(pdf_path: Path, text_out_dir: Path | None, full_text: str) -> None:
    if text_out_dir is None:
        return
    text_out_dir.mkdir(parents=True, exist_ok=True)
    (text_out_dir / f"{pdf_path.stem}.txt").write_text(full_text, encoding="utf-8")


def process_pdf(conn, pdf_path: Path, force: bool, text_out_dir: Path | None, ocr_workers: int) -> str:
    name = pdf_path.name
    if name in ingested_filenames(conn):
        if not force:
            return "skip (already ingested)"
        delete_existing(conn, name)
    full_text, page_count = pdf_pages_to_text(pdf_path, ocr_workers=max(1, ocr_workers))
    if not full_text:
        return "skip (no text extracted)"
    write_text_file(pdf_path, text_out_dir, full_text)
    chunks = chunk_text(full_text)
    if not chunks:
        return "skip (no chunks)"
    embeddings = [ollama_embed(c) for c in chunks]
    insert_document_and_chunks(conn, name, full_text, page_count, chunks, embeddings)
    return f"ok ({page_count}p, {len(chunks)} chunks)"


def iter_pdfs(pdf_dir: Path, limit: int | None) -> Iterable[Path]:
    pdfs = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
    return pdfs[:limit] if limit else pdfs


def preflight() -> None:
    if not OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")
    try:
        requests.get(f"{OPENROUTER_BASE_URL}/models", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, timeout=10).raise_for_status()
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        tags = {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        sys.exit(f"Preflight failed: {e}")
    if EMBED_MODEL not in tags and f"{EMBED_MODEL}:latest" not in tags:
        sys.exit(f"Embedding model `{EMBED_MODEL}` not pulled in Ollama. Run: ollama pull {EMBED_MODEL}")


def run_ingest(pdf_dir: Path, limit: int | None, force: bool, skip_preflight: bool, text_out_dir: str, ocr_workers: int) -> int:
    try:
        if not skip_preflight:
            preflight()
        conn = connect()
        try:
            all_pdfs = list(iter_pdfs(pdf_dir, limit))
            if not all_pdfs:
                print("[ingest] No PDFs found.")
                return 0
            ingested = set() if force else ingested_filenames(conn)
            pdfs = all_pdfs if force else [p for p in all_pdfs if p.name not in ingested]
            print(f"[ingest] Found {len(all_pdfs)} PDFs. Already ingested={len(all_pdfs)-len(pdfs)}. New to process={len(pdfs)}.")
            if not pdfs:
                return 0
            out_dir = Path(text_out_dir) if text_out_dir else None
            ok = err = skip = 0
            bar = tqdm(pdfs, unit="pdf")
            for pdf in bar:
                t0 = time.time()
                try:
                    status = process_pdf(conn, pdf, force=force, text_out_dir=out_dir, ocr_workers=max(1, ocr_workers))
                except Exception as e:
                    conn.rollback()
                    status = f"ERROR: {e}"
                    err += 1
                else:
                    if status.startswith("ok"):
                        ok += 1
                    elif status.startswith("skip"):
                        skip += 1
                bar.set_postfix_str(f"{pdf.name[:30]} -> {status} ({time.time() - t0:.1f}s)")
            print(f"\n[ingest] Done. ok={ok} skip={skip} err={err} total={len(pdfs)}")
            return 0 if err == 0 else 1
        finally:
            conn.close()
    except Exception as e:
        print(f"[ingest] Fatal error: {e}")
        return 1


def load_base_urls(base_url_args: list[str] | None, base_urls_file: str | None) -> list[str]:
    urls = list(base_url_args or [])
    if base_urls_file:
        for line in Path(base_urls_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out if out else list(DEFAULT_BASE_URLS)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", action="append", help="Seed base URL (repeatable). Defaults to NJ ethics + commissioner pages")
    ap.add_argument("--base-urls-file", help="Text file with one seed URL per line")
    ap.add_argument("--pdf-dir", default="pdf_files")
    ap.add_argument("--manifest-path", default="pdf_discovery_manifest.json")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--same-domain-only", action="store_true", default=True)
    ap.add_argument("--allow-cross-domain", action="store_true")
    ap.add_argument("--include-pattern")
    ap.add_argument("--exclude-pattern")
    ap.add_argument(
        "--exclude-prefix",
        action="append",
        default=None,
        help="URL prefix to exclude from crawling/discovery (repeatable).",
    )
    ap.add_argument(
        "--no-default-exclusions",
        action="store_true",
        help="Disable default excluded prefixes.",
    )
    ap.add_argument("--request-delay", type=float, default=0.8)
    ap.add_argument("--request-jitter", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--respect-retry-after", action="store_true", default=True)
    ap.add_argument("--user-agent", default="nj-sync-ingest/1.0 (+https://github.com/)")
    ap.add_argument("--max-pages", type=int, default=500, help="Safety cap for crawled HTML pages (0 disables cap)")
    ap.add_argument("--progress-every", type=int, default=25, help="Print crawl progress every N processed pages (0 disables)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--print-discovered", action="store_true", help="Print discovered PDF URLs")
    ap.add_argument("--print-new-only", action="store_true", help="When printing, only show discovered PDFs considered new")
    ap.add_argument(
        "--new-check-scope",
        choices=["downloaded", "ingested", "both"],
        default="downloaded",
        help="How to determine 'new' discovered PDFs",
    )
    ap.add_argument("--discovered-output", help="Write discovered list to file (.txt or .json)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--text-out-dir", default=TEXT_OUT_DIR)
    ap.add_argument("--ocr-workers", type=int, default=OCR_WORKERS)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    same_domain_only = False if args.allow_cross_domain else args.same_domain_only

    if args.ingest_only:
        raise SystemExit(run_ingest(pdf_dir, args.limit, args.force, args.skip_preflight, args.text_out_dir, args.ocr_workers))

    base_urls = load_base_urls(args.base_url, args.base_urls_file)
    print("[crawl] seed URLs:")
    for u in base_urls:
        print(f"  - {u}")

    client = PoliteHttpClient(
        request_delay=args.request_delay,
        request_jitter=args.request_jitter,
        timeout=args.timeout,
        max_retries=args.max_retries,
        user_agent=args.user_agent,
        respect_retry_after=args.respect_retry_after,
    )

    excluded_prefixes = [] if args.no_default_exclusions else list(DEFAULT_EXCLUDED_PREFIXES)
    if args.exclude_prefix:
        excluded_prefixes.extend(args.exclude_prefix)
    print("[crawl] excluded prefixes:")
    for p in excluded_prefixes:
        print(f"  - {p}")

    pdf_urls, source_map = crawl_for_pdfs(
        client,
        base_urls,
        max(0, args.max_depth),
        same_domain_only,
        args.include_pattern,
        args.exclude_pattern,
        excluded_prefixes,
        max_pages=max(0, args.max_pages),
        progress_every=max(0, args.progress_every),
    )
    print(f"[crawl] discovered {len(pdf_urls)} PDF URLs")
    write_manifest(Path(args.manifest_path), pdf_urls, source_map)

    if args.print_discovered or args.discovered_output:
        rows = discovered_output_rows(
            pdf_urls=pdf_urls,
            pdf_dir=pdf_dir,
            print_new_only=args.print_new_only,
            new_check_scope=args.new_check_scope,
        )
        if args.print_discovered:
            for r in rows:
                print(f"[discover] {r['filename']} :: {r['pdf_url']}")
            print(f"[discover] printed {len(rows)} items")
        if args.discovered_output:
            out = Path(args.discovered_output)
            write_discovered_output(out, rows)
            print(f"[discover] wrote {len(rows)} items to {out}")

    if args.dry_run:
        return

    downloaded, skipped, failed = download_pdfs(client, pdf_urls, pdf_dir)
    print(f"[download] downloaded={downloaded} skipped={skipped} failed={failed}")
    if args.download_only:
        return

    raise SystemExit(run_ingest(pdf_dir, args.limit, args.force, args.skip_preflight, args.text_out_dir, args.ocr_workers))


if __name__ == "__main__":
    main()
