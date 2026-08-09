# ICML 2026 Reproduction Paper Bundle

This repository builds a downloadable archive of the **official 200 arXiv-backed papers** listed in the Hugging Face `ICML-2026-agent-repro/challenge` Space's `curated.json` file.

## Why the PDFs are published as a GitHub Release

GitHub's normal Git repository storage is not suitable for a multi-hundred-paper PDF bundle because ordinary repository files have strict per-file size limits. Instead, this repository keeps the reproducible build recipe in Git and publishes the generated paper archive as a **GitHub Release asset**.

## Bundle contents

The generated archive contains:

- `papers/` — one PDF per unique arXiv ID
- `manifest.csv` — all curated challenge records, including OpenReview ID, arXiv ID, award field, and PDF URL
- `source_curated.json` — the exact Hugging Face curated manifest used for the build
- `README.txt` — bundle provenance and reconstruction notes
- `SHA256SUMS` — checksums for published assets

Duplicate arXiv IDs in the challenge manifest are downloaded only once, while all original challenge records remain in `manifest.csv`.

## Build / download

The workflow `.github/workflows/build-paper-bundle.yml` runs automatically when the bundling code changes and can also be triggered manually with **Actions → Build ICML 2026 curated paper bundle → Run workflow**.

It downloads the current official curated manifest from:

`https://huggingface.co/spaces/ICML-2026-agent-repro/challenge/raw/main/curated.json`

and expects exactly **200 challenge records**. The build stops if that count changes, preventing a silent scope change.

The finished archive is published under the GitHub Release tag:

`icml-2026-curated-papers`

If the archive exceeds GitHub's practical single-release-asset threshold, the workflow splits it into numbered parts and includes reassembly instructions in the release notes.

## Rebuild locally

```bash
python scripts/build_icml_paper_bundle.py --output dist
```

The script uses Python's standard library plus command-line `curl` and `zip`.

## Scope note

The challenge website can display a much larger ICML 2026 paper catalogue. This bundle intentionally targets the Space's **curated 200-paper arXiv-backed reproduction set**, because those entries have stable paper identifiers that can be downloaded reproducibly.
