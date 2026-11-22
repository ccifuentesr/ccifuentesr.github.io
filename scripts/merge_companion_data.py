#!/usr/bin/env python3
"""Merge instrument-specific CCF products into one chronological sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--fies", type=Path, required=True)
parser.add_argument("--hermes", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

fies = json.loads(args.fies.read_text())
hermes = json.loads(args.hermes.read_text())

for epoch in fies["epochs"]:
    epoch.setdefault("instrument", "FIES")
for epoch in hermes["epochs"]:
    epoch["instrument"] = "HERMES"

visible_epochs = {
    "2022-04-27T23:37",
    "2022-05-23T23:29",
    "2023-04-22T23:11",
    "2024-02-23T04:28",
    "2024-04-14T00:47",
    "2024-04-15T00:24",
    "2024-04-15T23:58",
    "2024-04-17T00:16",
}

combined = {
    "target": "RX J1417.5+0233",
    "frame": "heliocentric",
    "mask": fies["mask"],
    "wavelength_nm": fies["wavelength_nm"],
    "quality_thresholds": {"FIES": 10, "HERMES": 8},
    "epochs": sorted(
        [
            epoch
            for epoch in [*fies["epochs"], *hermes["epochs"]]
            if epoch["date"][:16] in visible_epochs
        ],
        key=lambda epoch: epoch["date"],
    ),
}

args.output.write_text(json.dumps(combined, indent=2) + "\n")
args.output.with_name("ccf-data.js").write_text(
    "window.hiddenCompanionsData = " + args.output.read_text().strip() + ";\n"
)
