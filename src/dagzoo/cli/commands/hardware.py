"""Hardware command handler."""

from __future__ import annotations

import argparse

from dagzoo.hardware import detect_hardware


def run_hardware_command(args: argparse.Namespace) -> int:
    """Execute the ``hardware`` command."""

    hw = detect_hardware(args.device)
    print(
        f"backend={hw.backend} device='{hw.device_name}' tier={hw.tier} "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e}"
    )
    return 0
