#!/usr/bin/env python3
"""Build a reproducible archive of the ICML 2026 Agent Repro curated papers."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = (
    "https://huggingface.co/spaces/ICML-2026-agent-repro/challenge/"
    "raw/main/curated.json"
)
EXPECTED_RECORDS = 200
MAX_RELEASE_ASSET_BYTES = 1_800 * 1024 * 1024
USER_AGENT = "ICML-2026-paper-bundle/1.0 (+GitHub Actions)"


def log(message: str) -> None:
    print(message, flush=True)


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def fetch_manifest(dest: Path) -> list[dict]:
    log(f"Fetching curated manifest: {SOURCE_URL}")
    payload = fetch_bytes(SOURCE_URL)
    dest.write_bytes(payload)
    records = json.loads(payload)
    if not isinstance(records, list):
        raise RuntimeError("curated.json is not a JSON list")
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} curated challenge records, got {len(records)}. "
            "Refusing to silently change bundle scope."
        )
    for i, record in enumerate(records, start=1):
        if not record.get("orid") or not record.get("alphaxiv"):
            raise RuntimeError(f"Record {i} is missing orid/alphaxiv: {record!r}")
    return records


def is_valid_pdf(path: Path) -> bool:
    try:
        if path.stat().st_size < 10_000:
            return False
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def curl_download(url: str, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.unlink(missing_ok=True)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "360",
        "-A",
        USER_AGENT,
        "-o",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(destination)


def download_one(arxiv_id: str, papers_dir: Path) -> tuple[str, int]:
    destination = papers_dir / f"arxiv_{arxiv_id}.pdf"
    if is_valid_pdf(destination):
        return arxiv_id, destination.stat().st_size

    urls = [
        f"https://arxiv.org/pdf/{arxiv_id}",
        f"https://export.arxiv.org/pdf/{arxiv_id}",
    ]
    last_error: Exception | None = None
    for attempt in range(1, 6):
        for url in urls:
            try:
                curl_download(url, destination)
                if not is_valid_pdf(destination):
                    raise RuntimeError("downloaded file is not a valid PDF")
                return arxiv_id, destination.stat().st_size
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_error = exc
                destination.unlink(missing_ok=True)
        if attempt < 5:
            time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(f"Failed to download arXiv:{arxiv_id}: {last_error}")


def write_manifest_csv(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "openreview_id", "arxiv_id", "award", "pdf_url"],
        )
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            arxiv_id = record["alphaxiv"]
            writer.writerow(
                {
                    "index": index,
                    "openreview_id": record["orid"],
                    "arxiv_id": arxiv_id,
                    "award": record.get("award", ""),
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                }
            )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_zip(bundle_dir: Path, release_dir: Path) -> list[Path]:
    archive = release_dir / "ICML-2026-curated-papers.zip"
    archive.unlink(missing_ok=True)

    # PDFs are already compressed; store mode avoids wasting CPU while preserving a ZIP package.
    subprocess.run(
        ["zip", "-0", "-q", "-r", str(archive), "."],
        cwd=bundle_dir,
        check=True,
    )

    if archive.stat().st_size <= MAX_RELEASE_ASSET_BYTES:
        return [archive]

    prefix = release_dir / "ICML-2026-curated-papers.zip.part-"
    subprocess.run(
        ["split", "-b", "1800M", "-d", "-a", "3", str(archive), str(prefix)],
        check=True,
    )
    archive.unlink()
    return sorted(release_dir.glob("ICML-2026-curated-papers.zip.part-*"))


def write_bundle_readme(
    path: Path,
    record_count: int,
    unique_count: int,
    duplicate_count: int,
    total_pdf_bytes: int,
) -> None:
    text = f"""ICML 2026 Agent Repro — Curated Paper Bundle
================================================

Source manifest:
{SOURCE_URL}

Challenge records: {record_count}
Unique arXiv PDFs: {unique_count}
Duplicate challenge mappings: {duplicate_count}
Total PDF bytes: {total_pdf_bytes}
Built UTC: {datetime.now(timezone.utc).isoformat()}

Contents
--------
papers/             Downloaded paper PDFs named by arXiv ID
manifest.csv         Original challenge mappings plus canonical PDF URLs
source_curated.json  Exact curated.json used for this build
README.txt           This file

Notes
-----
The Hugging Face challenge site can expose a larger ICML 2026 catalogue. This
bundle intentionally follows the Space's official curated 200-record arXiv-backed
set. Duplicate arXiv IDs are stored once in papers/, while every original challenge
record remains represented in manifest.csv.
"""
    path.write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    bundle_dir = output / "bundle"
    release_dir = output / "release"
    papers_dir = bundle_dir / "papers"

    if args.clean and output.exists():
        shutil.rmtree(output)
    papers_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = bundle_dir / "source_curated.json"
    records = fetch_manifest(source_manifest)
    write_manifest_csv(records, bundle_dir / "manifest.csv")

    arxiv_ids = [record["alphaxiv"] for record in records]
    counts = Counter(arxiv_ids)
    unique_ids = list(dict.fromkeys(arxiv_ids))
    duplicate_count = len(records) - len(unique_ids)

    log(
        f"Manifest OK: {len(records)} records, {len(unique_ids)} unique PDFs, "
        f"{duplicate_count} duplicate mappings."
    )

    failures: list[tuple[str, str]] = []
    completed: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, arxiv_id, papers_dir): arxiv_id
            for arxiv_id in unique_ids
        }
        total = len(futures)
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            arxiv_id = futures[future]
            try:
                _, size = future.result()
                completed[arxiv_id] = size
                log(f"[{n:03d}/{total:03d}] OK arXiv:{arxiv_id} ({size / 1024 / 1024:.1f} MiB)")
            except Exception as exc:  # noqa: BLE001 - aggregate build failures
                failures.append((arxiv_id, str(exc)))
                log(f"[{n:03d}/{total:03d}] FAIL arXiv:{arxiv_id}: {exc}")

    if failures:
        failure_path = output / "download_failures.txt"
        failure_path.write_text(
            "\n".join(f"{arxiv_id}\t{error}" for arxiv_id, error in failures) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"{len(failures)} paper downloads failed. See {failure_path}. "
            "Archive/release creation aborted to avoid publishing an incomplete bundle."
        )

    total_pdf_bytes = sum(completed.values())
    write_bundle_readme(
        bundle_dir / "README.txt",
        len(records),
        len(unique_ids),
        duplicate_count,
        total_pdf_bytes,
    )

    archive_assets = make_zip(bundle_dir, release_dir)

    # Copy provenance files next to the archive for quick inspection without unzipping.
    shutil.copy2(bundle_dir / "manifest.csv", release_dir / "manifest.csv")
    shutil.copy2(source_manifest, release_dir / "source_curated.json")

    checksum_targets = [*archive_assets, release_dir / "manifest.csv", release_dir / "source_curated.json"]
    checksum_path = release_dir / "SHA256SUMS"
    with checksum_path.open("w", encoding="utf-8") as f:
        for path in checksum_targets:
            f.write(f"{sha256(path)}  {path.name}\n")

    if len(archive_assets) == 1:
        archive_note = (
            "The full paper collection is contained in the single asset "
            f"`{archive_assets[0].name}`."
        )
    else:
        archive_note = (
            "The ZIP exceeded the safe single-asset threshold and was split into numbered parts. "
            "Reassemble it with:\n\n"
            "```bash\n"
            "cat ICML-2026-curated-papers.zip.part-* > ICML-2026-curated-papers.zip\n"
            "unzip ICML-2026-curated-papers.zip\n"
            "```"
        )

    release_notes = f"""# ICML 2026 curated paper bundle

Built from the Hugging Face `ICML-2026-agent-repro/challenge` Space's official
`curated.json` manifest.

- Challenge records: **{len(records)}**
- Unique arXiv PDFs: **{len(unique_ids)}**
- Duplicate mappings preserved in manifest: **{duplicate_count}**
- PDF payload: **{total_pdf_bytes / 1024 / 1024 / 1024:.2f} GiB**

{archive_note}

Use `SHA256SUMS` to verify downloaded assets. `manifest.csv` and
`source_curated.json` are also attached separately for provenance.
"""
    (release_dir / "RELEASE_NOTES.md").write_text(release_notes, encoding="utf-8")

    log("Build complete. Release assets:")
    for path in sorted(release_dir.iterdir()):
        log(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.2f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist", help="Build output directory")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent PDF downloads")
    parser.add_argument("--clean", action="store_true", help="Remove output directory first")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        build(parse_args())
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
