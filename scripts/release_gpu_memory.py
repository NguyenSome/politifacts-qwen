"""Utility to release GPU memory held by the current Python process."""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Iterable

import torch


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release GPU memory for the current process")
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device identifier (default: cuda). Use cuda:0, cuda:1, etc. If the device is unavailable the script exits gracefully.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print memory summary before and after cleanup.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is not available in this environment; nothing to release.")
        return 0

    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as err:
        print(f"Invalid device '{args.device}': {err}")
        return 1

    current_device = torch.cuda.current_device()
    target_idx = device.index if device.index is not None else current_device

    try:
        torch.cuda.set_device(target_idx)
    except RuntimeError as err:
        print(f"Unable to select CUDA device {target_idx}: {err}")
        return 1

    if args.summary:
        print("Before cleanup:\n")
        print(torch.cuda.memory_summary(device=target_idx, abbreviated=True))

    # Collect Python references and release cached allocator blocks.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    if args.summary:
        print("\nAfter cleanup:\n")
        print(torch.cuda.memory_summary(device=target_idx, abbreviated=True))
    else:
        print(
            f"Released cached memory on CUDA device {target_idx}.\n"
            "(Note: allocated tensors still referenced by your program will keep their memory.)"
        )

    # Restore the original device selection.
    torch.cuda.set_device(current_device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
