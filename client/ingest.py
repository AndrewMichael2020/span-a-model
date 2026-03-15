#!/usr/bin/env python3
"""
ingest.py — MacBook CLI prototype for span-a-model.

Scans a folder, resizes images, uploads to GCS, triggers the Phi-4 VM,
waits for results, and saves them as JSON files locally.

Usage:
  python ingest.py /path/to/photos --out ./results
  python ingest.py /path/to/photos --think --out ./results --no-cleanup
  python ingest.py --job-id job_20260315_123456_abc123  # resume / fetch results

Required env vars:
  PHI4_GCP_PROJECT   GCP project ID
  PHI4_BATCH_BUCKET  GCS bucket (created by terraform)
  PHI4_PUBSUB_TOPIC  Pub/Sub topic (default: phi4-job-requests)

Or put them in a .env file in this directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Load .env if present (no hard dependency — only used if dotenv is installed)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from phi4_client import Phi4Client

# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".webp", ".bmp"}

log = logging.getLogger("ingest")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Image discovery ───────────────────────────────────────────────────────────

def find_images(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        paths = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


# ── Progress bar (no external deps) ──────────────────────────────────────────

def progress_bar(done: int, total: int, width: int = 40, label: str = "") -> str:
    if total == 0:
        return "[empty]"
    filled = int(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * done // total
    return f"[{bar}] {done}/{total} ({pct}%) {label}"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest images into the span-a-model Phi-4 batch inference service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a folder (fast mode, no CoT):
  python ingest.py ~/Pictures/2024 --out ./results

  # Use thinking/CoT mode for complex documents:
  python ingest.py ~/Scans --think --out ./results

  # Resume / fetch results for an existing job:
  python ingest.py --job-id job_20260315_123456_abc123 --out ./results

  # Dry run (no upload):
  python ingest.py ~/Pictures --dry-run
        """,
    )

    parser.add_argument("folder", nargs="?", type=Path, help="Folder of images to process")
    parser.add_argument("--job-id", help="Resume or fetch results for an existing job ID")
    parser.add_argument("--out", type=Path, default=Path("./results"), help="Output directory for result JSONs (default: ./results)")
    parser.add_argument("--think", action="store_true", help="Use CoT reasoning mode (slower, more accurate)")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep images in GCS after processing")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan folder recursively")
    parser.add_argument("--max-px", type=int, default=1200, help="Max image dimension before upload (default: 1200)")
    parser.add_argument("--timeout", type=int, default=7200, help="Max seconds to wait for results (default: 7200)")
    parser.add_argument("--dry-run", action="store_true", help="Find images and show plan, but don't upload or trigger")
    parser.add_argument("-v", "--verbose", action="store_true")

    # GCP overrides (falls back to env vars)
    parser.add_argument("--project", help="GCP project ID (overrides PHI4_GCP_PROJECT)")
    parser.add_argument("--bucket", help="GCS bucket (overrides PHI4_BATCH_BUCKET)")
    parser.add_argument("--topic", help="Pub/Sub topic (overrides PHI4_PUBSUB_TOPIC)")

    return parser.parse_args()


def check_env(args: argparse.Namespace) -> None:
    """Apply CLI overrides to env vars so Phi4Client.from_env() picks them up."""
    if args.project:
        os.environ["PHI4_GCP_PROJECT"] = args.project
    if args.bucket:
        os.environ["PHI4_BATCH_BUCKET"] = args.bucket
    if args.topic:
        os.environ["PHI4_PUBSUB_TOPIC"] = args.topic

    missing = [v for v in ("PHI4_GCP_PROJECT", "PHI4_BATCH_BUCKET") if not os.environ.get(v)]
    if missing:
        print(f"\n❌  Missing required environment variables: {', '.join(missing)}")
        print("    Set them in .env or pass --project / --bucket\n")
        sys.exit(1)


def save_results(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        asset_id = r.get("asset_id", "unknown")
        out_path = out_dir / f"{asset_id}.json"
        out_path.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\n✅  {len(results)} results saved to {out_dir}/")


def print_summary(results: list[dict]) -> None:
    ok  = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]

    print(f"\n{'─'*60}")
    print(f"  Processed:  {len(ok)} succeeded,  {len(err)} failed")
    if err:
        print(f"\n  Failed assets:")
        for r in err:
            print(f"    • {r['asset_id']}: {r.get('error', '?')}")

    if ok:
        # Show a sample result
        sample = ok[0]
        print(f"\n  Sample result ({sample['asset_id']}):")
        print(f"    scene:   {sample.get('scene', 'n/a')}")
        print(f"    tags:    {', '.join(sample.get('tags', [])[:5])}")
        ocr = (sample.get('ocr_text') or '')[:120]
        if ocr:
            print(f"    ocr:     {ocr}…" if len(ocr) == 120 else f"    ocr:     {ocr}")
        summary = (sample.get('summary') or '')[:200]
        if summary:
            print(f"    summary: {summary}")
    print(f"{'─'*60}\n")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # ── Resume mode: just fetch results for existing job ──────────────────────
    if args.job_id and not args.folder:
        check_env(args)
        client = Phi4Client.from_env()
        client.max_px = args.max_px

        print(f"\n⏳  Waiting for job {args.job_id}…\n")
        try:
            results = client.wait_for_results(
                args.job_id,
                timeout_s=args.timeout,
                on_progress=lambda d, t: print(f"\r  {progress_bar(d, t)}", end="", flush=True),
            )
        except TimeoutError as e:
            print(f"\n⏰  {e}")
            sys.exit(1)

        save_results(results, args.out)
        print_summary(results)
        return

    # ── Normal mode: ingest a folder ──────────────────────────────────────────
    if not args.folder:
        print("❌  Provide a folder path or --job-id")
        sys.exit(1)

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"❌  Not a directory: {folder}")
        sys.exit(1)

    images = find_images(folder, args.recursive)
    if not images:
        print(f"❌  No images found in {folder}")
        sys.exit(1)

    thinking_mode = "think" if args.think else "nothink"

    print(f"\n{'═'*60}")
    print(f"  span-a-model — Phi-4-Reasoning-Vision batch inference")
    print(f"{'═'*60}")
    print(f"  Folder:        {folder}")
    print(f"  Images found:  {len(images)}")
    print(f"  Mode:          {thinking_mode} ({'CoT reasoning' if args.think else 'fast direct'})")
    print(f"  Max dimension: {args.max_px}px")
    print(f"  Output:        {args.out}")
    print(f"{'─'*60}\n")

    if args.dry_run:
        print("  DRY RUN — showing first 10 images:\n")
        for p in images[:10]:
            print(f"    {p.name}")
        if len(images) > 10:
            print(f"    … and {len(images) - 10} more")
        print("\n  (no upload, no job submitted)\n")
        return

    check_env(args)
    client = Phi4Client.from_env()
    client.max_px = args.max_px

    # Upload + submit
    print("  Uploading images to GCS…\n")
    start = time.time()
    last_print = [0]

    def upload_progress(done: int, total: int) -> None:
        now = time.time()
        if done == total or now - last_print[0] > 2:
            print(f"\r  {progress_bar(done, total, label='uploaded')}", end="", flush=True)
            last_print[0] = now

    try:
        job_id = client.submit_batch(
            image_paths=images,
            thinking_mode=thinking_mode,
            job_id=args.job_id,
            on_progress=upload_progress,
        )
    except Exception as e:
        print(f"\n❌  Failed to submit batch: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n\n  ✅  Job submitted: {job_id}  ({elapsed:.1f}s upload)")
    print(f"\n  VM is starting… (first boot takes ~5–8 min to load the model)")
    print(f"  To resume later if this terminal closes:\n")
    print(f"      python ingest.py --job-id {job_id} --out {args.out}\n")

    # Wait for results
    print("  Waiting for inference results…\n")
    try:
        results = client.wait_for_results(
            job_id,
            timeout_s=args.timeout,
            on_progress=lambda d, t: print(f"\r  {progress_bar(d, t, label='inferred')}", end="", flush=True),
        )
    except TimeoutError as e:
        print(f"\n⏰  {e}")
        print(f"  Fetch results later with:  python ingest.py --job-id {job_id} --out {args.out}")
        sys.exit(1)

    print()  # newline after progress bar

    save_results(results, args.out)
    print_summary(results)

    # Cleanup
    if not args.no_cleanup:
        print("  Cleaning up GCS objects…")
        deleted = client.cleanup_job(job_id)
        print(f"  Deleted {deleted} GCS objects\n")
    else:
        print(f"  --no-cleanup: GCS objects retained at gs://{client.batch_bucket}/{job_id}/\n")


if __name__ == "__main__":
    main()
