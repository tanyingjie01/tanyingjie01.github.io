#!/usr/bin/env python3
"""Use directory URLs for generated blog pages instead of /index.html URLs."""

import os
import re
from pathlib import Path


project_dir = Path(__file__).resolve().parents[1]
output_dir = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site"))
if not output_dir.is_absolute():
    output_dir = project_dir / output_dir

blog_index = re.compile(
    r'(?<![\w./:-])'
    r'(?:https://tanyingjie01\.github\.io/|(?:\.\./)+|\./|/)?'
    r'(?:zh/)?blog/(?:[^/"\'<>\s?#]+/)*index\.html(?=["\'<>\s?#]|$)'
)

for extension in ("*.html", "*.xml", "*.json"):
    for path in output_dir.rglob(extension):
        original = path.read_text(encoding="utf-8")
        updated = blog_index.sub(lambda match: match.group()[:-len("index.html")], original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
