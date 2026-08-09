#!/usr/bin/env python3
"""Build archival and Teams-lite bundles for the ICML 2026 Agent Repro curated papers."""

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
TEAMS_LITE_TARGET_BYTES = 800 * 1024 * 1024
TEAMS_LITE_PAYLOAD_TARGET_BYTES = 790 * 1024 * 1024
USER_AGENT = "ICML-2026-paper-bundle/2.0 (+GitHub Actions)"

# First pass is intentionally readable for papers. If the payload remains above
# target, only the largest remaining PDFs are re-rendered at lower image DPI.
LITE_PROFILES = [
    {"name": "150dpi-q82", "dpi": 150, "jpeg_q": 82, "mono_dpi": 300},
    {"name": "120dpi-q80", "dpi": 120, "jpeg_q": 80, "mono_dpi": 240},
    {"name": "96dpi-q76", "dpi": 96, "jpeg_q": 76, "mono_dpi": 192},
]


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
            time.sleep(min(30, 2**attempt))

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


def zip_directory(bundle_dir: Path, archive: Path) -> None:
    archive.unlink(missing_ok=True)
    subprocess.run(
        ["zip", "-9", "-q", "-r", str(archive), "."],
        cwd=bundle_dir,
        check=True,
    )


def make_archival_zip(bundle_dir: Path, release_dir: Path) -> list[Path]:
    archive = release_dir / "ICML-2026-curated-papers.zip"
    zip_directory(bundle_dir, archive)

    if archive.stat().st_size <= MAX_RELEASE_ASSET_BYTES:
        return [archive]

    prefix = release_dir / "ICML-2026-curated-papers.zip.part-"
    subprocess.run(
        ["split", "-b", "1800M", "-d", "-a", "3", str(archive), str(prefix)],
        check=True,
    )
    archive.unlink()
    return sorted(release_dir.glob("ICML-2026-curated-papers.zip.part-*"))


def ghostscript_optimize(
    source: Path,
    destination: Path,
    *,
    dpi: int,
    jpeg_q: int,
    mono_dpi: int,
) -> tuple[int, bool]:
    """Create a reading-optimized PDF. Return (bytes, accepted).

    Text and vector content remain vector where Ghostscript can preserve it.
    Raster images are downsampled/re-encoded. The candidate is accepted only
    when it is a valid PDF and smaller than the current lite copy.
    """
    candidate = destination.with_suffix(destination.suffix + ".candidate")
    candidate.unlink(missing_ok=True)

    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.6",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        "-dSAFER",
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        "-dEmbedAllFonts=true",
        "-dAutoRotatePages=/None",
        "-dDownsampleColorImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={dpi}",
        "-dColorImageDownsampleThreshold=1.0",
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dDownsampleGrayImages=true",
        "-dGrayImageDownsampleType=/Bicubic",
        f"-dGrayImageResolution={dpi}",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",
        f"-dMonoImageResolution={mono_dpi}",
        f"-dJPEGQ={jpeg_q}",
        f"-sOutputFile={candidate}",
        str(source),
    ]
    subprocess.run(cmd, check=True)

    if not is_valid_pdf(candidate):
        candidate.unlink(missing_ok=True)
        return destination.stat().st_size, False

    current_size = destination.stat().st_size
    candidate_size = candidate.stat().st_size
    if candidate_size >= current_size:
        candidate.unlink(missing_ok=True)
        return current_size, False

    candidate.replace(destination)
    return candidate_size, True


def copy_or_link(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def optimize_initial(
    arxiv_id: str,
    source: Path,
    destination: Path,
    profile: dict,
) -> dict:
    copy_or_link(source, destination)
    original_size = source.stat().st_size
    original_hash = sha256(source)

    try:
        lite_size, accepted = ghostscript_optimize(
            source,
            destination,
            dpi=profile["dpi"],
            jpeg_q=profile["jpeg_q"],
            mono_dpi=profile["mono_dpi"],
        )
        profile_name = profile["name"] if accepted else "original"
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not destroy archival build
        log(f"Lite optimization warning arXiv:{arxiv_id}: {exc}")
        copy_or_link(source, destination)
        lite_size = original_size
        accepted = False
        profile_name = "original"

    return {
        "arxiv_id": arxiv_id,
        "original_size": original_size,
        "lite_size": lite_size,
        "original_sha256": original_hash,
        "lite_sha256": sha256(destination),
        "optimized": accepted,
        "profile": profile_name,
    }


def build_teams_lite(
    unique_ids: list[str],
    archival_papers_dir: Path,
    lite_bundle_dir: Path,
    records: list[dict],
    source_manifest: Path,
    *,
    workers: int,
) -> tuple[Path, Path, int, list[dict]]:
    papers_dir = lite_bundle_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_manifest, lite_bundle_dir / "source_curated.json")
    write_manifest_csv(records, lite_bundle_dir / "manifest.csv")

    first_profile = LITE_PROFILES[0]
    stats_by_id: dict[str, dict] = {}
    log(
        f"Building Teams Lite first pass: {first_profile['name']} "
        f"with {workers} worker(s)."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for arxiv_id in unique_ids:
            source = archival_papers_dir / f"arxiv_{arxiv_id}.pdf"
            destination = papers_dir / f"arxiv_{arxiv_id}.pdf"
            future = pool.submit(
                optimize_initial,
                arxiv_id,
                source,
                destination,
                first_profile,
            )
            futures[future] = arxiv_id

        total = len(futures)
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            arxiv_id = futures[future]
            stats = future.result()
            stats_by_id[arxiv_id] = stats
            reduction = 100.0 * (
                1.0 - stats["lite_size"] / max(1, stats["original_size"])
            )
            log(
                f"[lite {n:03d}/{total:03d}] arXiv:{arxiv_id} "
                f"{stats['profile']} {stats['original_size']/1024/1024:.1f} -> "
                f"{stats['lite_size']/1024/1024:.1f} MiB ({reduction:.1f}% saved)"
            )

    def payload_size() -> int:
        return sum(stats_by_id[arxiv_id]["lite_size"] for arxiv_id in unique_ids)

    current_payload = payload_size()
    log(f"Teams Lite payload after first pass: {current_payload/1024/1024:.1f} MiB")

    for profile in LITE_PROFILES[1:]:
        if current_payload <= TEAMS_LITE_PAYLOAD_TARGET_BYTES:
            break

        log(
            f"Payload still above {TEAMS_LITE_PAYLOAD_TARGET_BYTES/1024/1024:.0f} MiB; "
            f"selectively applying {profile['name']} to largest PDFs."
        )
        candidates = sorted(
            unique_ids,
            key=lambda aid: stats_by_id[aid]["lite_size"],
            reverse=True,
        )
        for arxiv_id in candidates:
            if current_payload <= TEAMS_LITE_PAYLOAD_TARGET_BYTES:
                break

            source = archival_papers_dir / f"arxiv_{arxiv_id}.pdf"
            destination = papers_dir / f"arxiv_{arxiv_id}.pdf"
            before = destination.stat().st_size
            try:
                after, accepted = ghostscript_optimize(
                    source,
                    destination,
                    dpi=profile["dpi"],
                    jpeg_q=profile["jpeg_q"],
                    mono_dpi=profile["mono_dpi"],
                )
            except Exception as exc:  # noqa: BLE001
                log(f"Selective lite warning arXiv:{arxiv_id}: {exc}")
                continue

            if accepted and after < before:
                current_payload -= before - after
                stats = stats_by_id[arxiv_id]
                stats["lite_size"] = after
                stats["lite_sha256"] = sha256(destination)
                stats["optimized"] = True
                stats["profile"] = profile["name"]
                log(
                    f"  {arxiv_id}: {before/1024/1024:.1f} -> "
                    f"{after/1024/1024:.1f} MiB; payload "
                    f"{current_payload/1024/1024:.1f} MiB"
                )

    if current_payload > TEAMS_LITE_PAYLOAD_TARGET_BYTES:
        raise RuntimeError(
            "Teams Lite could not reach the safe payload target even after the "
            f"{LITE_PROFILES[-1]['name']} fallback. Payload is "
            f"{current_payload/1024/1024:.1f} MiB."
        )

    lite_manifest = lite_bundle_dir / "lite_manifest.csv"
    with lite_manifest.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "arxiv_id",
            "original_bytes",
            "lite_bytes",
            "reduction_percent",
            "original_sha256",
            "lite_sha256",
            "optimization_applied",
            "profile",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for arxiv_id in unique_ids:
            stats = stats_by_id[arxiv_id]
            writer.writerow(
                {
                    "arxiv_id": arxiv_id,
                    "original_bytes": stats["original_size"],
                    "lite_bytes": stats["lite_size"],
                    "reduction_percent": (
                        f"{100.0 * (1.0 - stats['lite_size'] / max(1, stats['original_size'])):.2f}"
                    ),
                    "original_sha256": stats["original_sha256"],
                    "lite_sha256": stats["lite_sha256"],
                    "optimization_applied": str(stats["optimized"]).lower(),
                    "profile": stats["profile"],
                }
            )

    original_total = sum(stats_by_id[aid]["original_size"] for aid in unique_ids)
    optimized_count = sum(1 for aid in unique_ids if stats_by_id[aid]["optimized"])
    readme = f"""ICML 2026 Agent Repro — TEAMS LITE Reading Edition
=====================================================

This is a space-saving reading copy derived from the archival bundle.

Target final ZIP size: <= {TEAMS_LITE_TARGET_BYTES / 1024 / 1024:.0f} MiB
Lite PDF payload: {current_payload / 1024 / 1024:.1f} MiB
Original PDF payload: {original_total / 1024 / 1024:.1f} MiB
Optimized PDFs: {optimized_count} / {len(unique_ids)}
Built UTC: {datetime.now(timezone.utc).isoformat()}

IMPORTANT
---------
This edition is NOT byte-identical to the source arXiv PDFs when
optimization_applied=true. Raster images may be downsampled/re-encoded to reduce
storage. Text and vector content are kept as vector content where Ghostscript can
preserve it.

Use ICML-2026-curated-papers.zip for archival/reproducibility work.
Use this TEAMS-LITE edition for convenient storage, sharing, and reading.

See lite_manifest.csv for per-paper source/lite SHA256 values, byte sizes, and
the optimization profile used.
"""
    (lite_bundle_dir / "README_TEAMS_LITE.txt").write_text(readme, encoding="utf-8")

    return lite_manifest, lite_bundle_dir / "README_TEAMS_LITE.txt", current_payload, [
        stats_by_id[aid] for aid in unique_ids
    ]


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

The ZIP uses maximum lossless DEFLATE compression. The PDFs themselves are kept
byte-for-byte unchanged; no image resampling or PDF recompression is performed.
"""
    path.write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    bundle_dir = output / "bundle"
    lite_bundle_dir = output / "lite_bundle"
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
                log(
                    f"[{n:03d}/{total:03d}] OK arXiv:{arxiv_id} "
                    f"({size / 1024 / 1024:.1f} MiB)"
                )
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

    archive_assets = make_archival_zip(bundle_dir, release_dir)

    lite_manifest, lite_readme, lite_payload_bytes, lite_stats = build_teams_lite(
        unique_ids,
        papers_dir,
        lite_bundle_dir,
        records,
        source_manifest,
        workers=args.lite_workers,
    )

    lite_archive = release_dir / "ICML-2026-curated-papers-TEAMS-LITE.zip"
    zip_directory(lite_bundle_dir, lite_archive)
    if lite_archive.stat().st_size > TEAMS_LITE_TARGET_BYTES:
        raise RuntimeError(
            "Teams Lite ZIP exceeded the hard target: "
            f"{lite_archive.stat().st_size / 1024 / 1024:.1f} MiB > "
            f"{TEAMS_LITE_TARGET_BYTES / 1024 / 1024:.0f} MiB."
        )

    shutil.copy2(bundle_dir / "manifest.csv", release_dir / "manifest.csv")
    shutil.copy2(source_manifest, release_dir / "source_curated.json")
    shutil.copy2(lite_manifest, release_dir / "lite_manifest.csv")
    shutil.copy2(lite_readme, release_dir / "README_TEAMS_LITE.txt")

    checksum_targets = [
        *archive_assets,
        lite_archive,
        release_dir / "manifest.csv",
        release_dir / "lite_manifest.csv",
        release_dir / "source_curated.json",
        release_dir / "README_TEAMS_LITE.txt",
    ]
    checksum_path = release_dir / "SHA256SUMS"
    with checksum_path.open("w", encoding="utf-8") as f:
        for path in checksum_targets:
            f.write(f"{sha256(path)}  {path.name}\n")

    if len(archive_assets) == 1:
        archive_note = (
            "The archival source collection is in "
            f"`{archive_assets[0].name}` using maximum lossless ZIP/DEFLATE compression."
        )
    else:
        archive_note = (
            "The archival ZIP exceeded the safe single-asset threshold and was split into "
            "numbered parts. Reassemble it with:\n\n"
            "```bash\n"
            "cat ICML-2026-curated-papers.zip.part-* > ICML-2026-curated-papers.zip\n"
            "unzip ICML-2026-curated-papers.zip\n"
            "```"
        )

    optimized_count = sum(1 for stats in lite_stats if stats["optimized"])
    total_lite_saved = total_pdf_bytes - lite_payload_bytes
    release_notes = f"""# ICML 2026 curated paper bundle

Built from the Hugging Face `ICML-2026-agent-repro/challenge` Space's official
`curated.json` manifest.

- Challenge records: **{len(records)}**
- Unique arXiv PDFs: **{len(unique_ids)}**
- Duplicate mappings preserved in manifest: **{duplicate_count}**
- Original PDF payload: **{total_pdf_bytes / 1024 / 1024 / 1024:.2f} GiB**
- Archival mode: **byte-identical source PDFs + maximum lossless ZIP/DEFLATE**
- Teams Lite PDF payload: **{lite_payload_bytes / 1024 / 1024:.1f} MiB**
- Teams Lite ZIP: **{lite_archive.stat().st_size / 1024 / 1024:.1f} MiB**
- Teams Lite optimized PDFs: **{optimized_count}/{len(unique_ids)}**
- Teams Lite PDF bytes saved: **{total_lite_saved / 1024 / 1024:.1f} MiB**

{archive_note}

## Which download should I use?

- **Research / reproduction:** `ICML-2026-curated-papers.zip`
  - source arXiv PDFs are preserved byte-for-byte.
- **Teams / reading:** `ICML-2026-curated-papers-TEAMS-LITE.zip`
  - final archive is required to stay at or below **800 MiB**.
  - raster images may be downsampled/re-encoded; text/vector content is preserved
    where the PDF rendering pipeline can preserve it.
  - do not use this edition when byte-identical source artifacts are required.

`lite_manifest.csv` records original/lite sizes, SHA256 hashes, and the profile
used for every unique paper. Use `SHA256SUMS` to verify release assets.
"""
    (release_dir / "RELEASE_NOTES.md").write_text(release_notes, encoding="utf-8")

    log("Build complete. Release assets:")
    for path in sorted(release_dir.iterdir()):
        log(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.2f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist", help="Build output directory")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent PDF downloads")
    parser.add_argument(
        "--lite-workers",
        type=int,
        default=2,
        help="Concurrent Ghostscript jobs for Teams Lite",
    )
    parser.add_argument("--clean", action="store_true", help="Remove output directory first")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        build(parse_args())
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
