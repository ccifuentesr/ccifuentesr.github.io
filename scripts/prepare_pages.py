#!/usr/bin/env python3
"""Stage public static assets only, excluding source services and local config."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / '_site'
if TARGET.exists():
    shutil.rmtree(TARGET)
TARGET.mkdir()
for source in ROOT.iterdir():
    if source.is_file() and source.suffix.lower() in {'.html', '.js', '.png', '.svg', '.ico', '.webmanifest'}:
        shutil.copy2(source, TARGET / source.name)
for name in ('assets', 'media'):
    shutil.copytree(ROOT / name, TARGET / name, ignore=shutil.ignore_patterns('.*', '__pycache__'))
(TARGET / '.nojekyll').touch()
