"""
fix_reports.py
--------------
One-time script to patch existing HTML reports with:
  - **bold** → <strong>bold</strong>
  - Numbered section headings styled with .narrative-heading
  - Updated CSS for .narrative-heading

Run once:
    python fix_reports.py
"""

import re
from pathlib import Path

NARRATIVE_HEADING_CSS = """\
  .narrative-heading { display: block; font-size: 16px; font-weight: bold;
                        color: #1D9E75; margin-top: 24px; margin-bottom: 6px;
                        letter-spacing: 0.4px; border-bottom: 1px solid #d4ede6;
                        padding-bottom: 4px; }
  .narrative-heading:first-child { margin-top: 4px; }"""


def patch_narrative(html: str) -> str:
    # 1. Inject .narrative-heading CSS before </style>
    if ".narrative-heading" not in html:
        html = html.replace("</style>", f"{NARRATIVE_HEADING_CSS}\n</style>", 1)

    # 2. Extract the narrative div content, patch it, put it back
    start_tag = 'class="narrative">'
    start      = html.find(start_tag)
    if start == -1:
        return html
    content_start = start + len(start_tag)
    end           = html.find("</div>", content_start)
    if end == -1:
        return html

    narrative = html[content_start:end]

    # **bold** → <strong>bold</strong>
    narrative = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', narrative)

    # Numbered section headings → styled span
    narrative = re.sub(
        r'(\d\.\s+[A-Z][A-Z\s&]+?)(\s*<br>)',
        r'<span class="narrative-heading">\1</span>\2',
        narrative,
    )

    return html[:content_start] + narrative + html[end:]


def main():
    reports = list(Path("outputs").glob("*_report.html"))
    if not reports:
        print("No report HTML files found in outputs/")
        return

    for path in reports:
        original = path.read_text(encoding="utf-8")
        patched  = patch_narrative(original)
        if patched != original:
            path.write_text(patched, encoding="utf-8")
            print(f"Patched: {path.name}")
        else:
            print(f"No changes: {path.name}")


if __name__ == "__main__":
    main()
