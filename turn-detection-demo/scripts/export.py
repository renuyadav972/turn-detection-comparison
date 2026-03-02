#!/usr/bin/env python3
"""
Static HTML Exporter
====================
Reads session JSON files and bakes them into a self-contained HTML file
that can be deployed to GitHub Pages, Netlify, etc.

Usage:
    python scripts/export.py data/sessions/*.json -o comparison.html
    python scripts/export.py data/sessions/abc.json data/sessions/def.json -o comparison.html
"""

import argparse
import json
import sys
from pathlib import Path


def load_sessions(paths: list[str]) -> list[dict]:
    sessions = []
    for p in paths:
        try:
            with open(p) as f:
                sessions.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {p}: {e}", file=sys.stderr)
    return sessions


def build_html(sessions: list[dict]) -> str:
    # Read the dashboard HTML template
    template_path = Path(__file__).parent.parent / "static" / "dashboard.html"
    template = template_path.read_text()

    # Inject session data as a global variable before the closing </script> tag.
    # The dashboard.html already checks for window.SESSION_DATA.
    data_json = json.dumps(sessions, indent=2)
    injection = f"window.SESSION_DATA = {data_json};\n"

    # Replace the init section to use embedded data
    # Insert the data right after the <script> opening tag
    template = template.replace(
        "<script>\n// \u2500\u2500\u2500 State",
        f"<script>\n{injection}\n// \u2500\u2500\u2500 State",
    )

    # Remove server polling (no API available in static mode)
    template = template.replace(
        "setInterval(loadSessions, 5000);",
        "// Static mode — no polling",
    )

    # Update the title
    template = template.replace(
        "<title>Turn Detection Metrics Dashboard</title>",
        "<title>Turn Detection Comparison</title>",
    )

    return template


def main():
    parser = argparse.ArgumentParser(description="Export session data to static HTML")
    parser.add_argument("files", nargs="+", help="Session JSON files")
    parser.add_argument("-o", "--output", default="comparison.html", help="Output HTML file")
    args = parser.parse_args()

    sessions = load_sessions(args.files)
    if not sessions:
        print("Error: no valid session files found.", file=sys.stderr)
        sys.exit(1)

    html = build_html(sessions)
    Path(args.output).write_text(html)
    print(f"Exported {len(sessions)} sessions to {args.output}")


if __name__ == "__main__":
    main()
