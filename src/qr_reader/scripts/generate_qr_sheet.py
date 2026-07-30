"""Generate a printable PDF sheet of QR codes for the human-made test dataset.

Produces: data/bench/qr_sheet.pdf
5 versions x 3 physical sizes = 15 QR codes, each at its specified size.
"""

from pathlib import Path

import cv2
from fpdf import FPDF

from qr_reader.synth.patch import generate_qr_patch

URL = "rikvoorhaar.com"
VERSIONS = [1, 3, 5, 10, 20]
SIZES_MM = [30, 60, 120]
DPI = 300
MM_PER_INCH = 25.4

OUT_DIR = Path("data/bench")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARGIN = 12
GAP = 6


def qr_to_file(version: int, size_mm: float) -> Path:
    px = int(round(size_mm / MM_PER_INCH * DPI))
    N = 17 + 4 * version
    ppm = int(round(px / N))
    patch, _ = generate_qr_patch(
        version=version, content=URL, ecl_str="L",
        ppm=ppm, quiet_zone_modules=4,
    )
    path = OUT_DIR / f"_tmp_qr_v{version}_s{size_mm}.png"
    cv2.imwrite(str(path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    return path


def add_title(pdf, label: str, size_mm: int):
    pdf.set_font("Helvetica", size=10)
    pdf.text(x=MARGIN, y=8, text=f"QR Dataset - {URL}  [{label}]")
    pdf.set_font("Helvetica", size=7)
    pdf.text(x=MARGIN, y=13, text=f"Print at 300 DPI  |  {size_mm}mm squares  |  Laminate + cut out")


def main():
    pdf = FPDF(orientation="P", unit="mm", format="A4")

    # ── 30mm: landscape, 5 columns ──
    pdf.add_page("L")
    add_title(pdf, "Small", 30)
    page_w = 297
    usable = page_w - 2 * MARGIN
    col_w = (usable - (len(VERSIONS) - 1) * GAP) / len(VERSIONS)
    qr_w = min(30, col_w - 2)
    y = 20
    for ci, v in enumerate(VERSIONS):
        x = MARGIN + ci * (col_w + GAP) + (col_w - qr_w) / 2
        tmp = qr_to_file(v, qr_w)
        pdf.image(str(tmp), x=x, y=y, w=qr_w, h=qr_w)
        pdf.set_font("Helvetica", size=7)
        pdf.text(x=x, y=y + qr_w + 3, text=f"V{v}")

    # ── 60mm: landscape, 3 columns × 2 pages ──
    for page_start in range(0, len(VERSIONS), 3):
        pdf.add_page("L")
        add_title(pdf, f"Medium (page {pdf.page_no()})", 60)
        usable = 297 - 2 * MARGIN
        n_cols = min(3, len(VERSIONS) - page_start)
        col_w = (usable - (n_cols - 1) * GAP) / n_cols
        qr_w = min(60, col_w - 2)
        y = 20 + (col_w - qr_w) / 2
        for i in range(n_cols):
            v = VERSIONS[page_start + i]
            x = MARGIN + i * (col_w + GAP) + (col_w - qr_w) / 2
            tmp = qr_to_file(v, qr_w)
            pdf.image(str(tmp), x=x, y=y, w=qr_w, h=qr_w)
            pdf.set_font("Helvetica", size=7)
            pdf.text(x=x, y=y + qr_w + 3, text=f"V{v}")

    # ── 120mm: portrait, 1 per page ──
    for v in VERSIONS:
        pdf.add_page("P")
        add_title(pdf, f"Large - V{v}", 120)
        usable_w = 210 - 2 * MARGIN
        qr_w = min(120, usable_w - 10)
        x = MARGIN + (usable_w - qr_w) / 2
        y = 25
        tmp = qr_to_file(v, qr_w)
        pdf.image(str(tmp), x=x, y=y, w=qr_w, h=qr_w)
        pdf.set_font("Helvetica", size=12)
        pdf.text(x=x, y=y + qr_w + 6, text=f"Version {v}  |  {qr_w:.0f}mm  |  rikvoorhaar.com")

    out_path = OUT_DIR / "qr_sheet.pdf"
    pdf.output(str(out_path))

    for tmp in OUT_DIR.glob("_tmp_qr_*.png"):
        tmp.unlink()

    print(f"Saved {out_path}")
    print(f"  {len(VERSIONS)} versions x {len(SIZES_MM)} sizes = {len(VERSIONS)*len(SIZES_MM)} QR codes")
    print(f"  {pdf.page_no()} pages")
    for v in VERSIONS:
        print(f"    V={v:2d}   30mm  60mm  120mm")


if __name__ == "__main__":
    main()
