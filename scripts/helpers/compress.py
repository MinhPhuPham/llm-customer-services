# ===========================================================
# compress.py — Zstandard compression for OTA delivery
# ===========================================================
"""
Compresses exported model files with zstandard (level 19)
for efficient over-the-air delivery via CDN.
"""

import os

import zstandard as zstd


def compress_for_ota(file_paths):
    """
    Compress files with zstandard for OTA delivery.

    Args:
        file_paths: List of (label, path) tuples to compress.

    Returns:
        results: List of (label, original_mb, compressed_mb) tuples.
    """
    print("=" * 60)
    print("COMPRESSION (OTA)")
    print("=" * 60)

    cctx = zstd.ZstdCompressor(level=19)
    results = []

    for label, src_path in file_paths:
        if not os.path.exists(src_path):
            print(f"  SKIP: {label} — file not found")
            continue

        dst_path = src_path + '.zst'
        with open(src_path, 'rb') as f:
            raw = f.read()
        with open(dst_path, 'wb') as f:
            f.write(cctx.compress(raw))

        original_mb = len(raw) / 1e6
        compressed_mb = os.path.getsize(dst_path) / 1e6
        results.append((label, original_mb, compressed_mb))
        print(f"  {label}: {original_mb:.1f}MB → {compressed_mb:.1f}MB (zstd)")

    return results
