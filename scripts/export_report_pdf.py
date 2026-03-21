from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_pandoc(markdown_path: Path, html_path: Path, css_path: Path, resource_root: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pandoc",
        str(markdown_path),
        "--standalone",
        "--from",
        "markdown+yaml_metadata_block+pipe_tables-implicit_figures",
        "--to",
        "html5",
        "--mathml",
        "--embed-resources",
        "--resource-path",
        str(resource_root),
        "--css",
        css_path.name,
        "-o",
        str(html_path.name),
    ]
    subprocess.run(command, check=True, cwd=html_path.parent)


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.emulate_media(media="print")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                "<div style=\"width:100%; font-size:8px; color:#64748b; "
                "padding:0 12mm; text-align:center;\">"
                "<span class=\"pageNumber\"></span> / <span class=\"totalPages\"></span>"
                "</div>"
            ),
            margin={
                "top": "14mm",
                "right": "12mm",
                "bottom": "16mm",
                "left": "12mm",
            },
            prefer_css_page_size=True,
        )
        browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export markdown report to PDF with browser rendering.")
    parser.add_argument("--markdown", default="docs/Thesis/report.md", help="Path to input markdown report.")
    parser.add_argument("--pdf", default="docs/Thesis/report.pdf", help="Path to output PDF.")
    parser.add_argument(
        "--html",
        default="tmp/report_generation/report_print.html",
        help="Path to intermediate HTML file.",
    )
    parser.add_argument(
        "--css",
        default="tmp/report_generation/report_print.css",
        help="Path to print CSS file.",
    )
    parser.add_argument(
        "--resource-root",
        default=".",
        help="Root path for resolving local images and other markdown resources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown_path = Path(args.markdown).resolve()
    pdf_path = Path(args.pdf).resolve()
    html_path = Path(args.html).resolve()
    css_path = Path(args.css).resolve()
    resource_root = Path(args.resource_root).resolve()

    run_pandoc(markdown_path, html_path, css_path, resource_root)
    print_pdf(html_path, pdf_path)
    print(f"Exported PDF: {pdf_path}")


if __name__ == "__main__":
    main()
