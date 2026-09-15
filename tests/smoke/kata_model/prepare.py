"""Prepare a Docker build directory and verify the pinned official Qwen GGUF."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

REVISION = "13fb94bfda8c8cf22497dc57b78f391a9acb426a"
FILENAME = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
SHA256 = "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "model.gguf"
    if not target.exists():
        url = ("https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/"
               f"{REVISION}/{FILENAME}")
        partial = target.with_suffix(".part")
        print("Downloading the official Qwen GGUF (4.68 GB)", flush=True)
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
        with partial.open("rb") as downloaded:
            if hashlib.file_digest(downloaded, "sha256").hexdigest() != SHA256:
                raise RuntimeError("Model SHA256 mismatch; refusing to use the download")
        partial.rename(target)
    with target.open("rb") as downloaded:
        if hashlib.file_digest(downloaded, "sha256").hexdigest() != SHA256:
            raise RuntimeError("Existing model SHA256 mismatch")
    for name in ("Dockerfile", ".dockerignore", "mini-config.yaml", "clamp.py", "clamp_checks.py"):
        shutil.copyfile(Path(__file__).parent / name, args.output / name)
    print(f"Verified model and build context ready: {args.output}", flush=True)


if __name__ == "__main__":
    main()
