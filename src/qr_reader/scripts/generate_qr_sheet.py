"""Generate a printable PDF sheet of QR codes for the human-made test dataset.

Produces: data/bench/qr_sheet.pdf
5 versions x 3 physical sizes = 15 QR codes, grid layout with labels.
All QR codes encode the same URL.
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


def main():
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    margin_left = 15
    margin_top = 20
    col_gap = 8
    row_gap = 15
    page_w = 210
    cell_w = (page_w - 2 * margin_left - 2 * col_gap) / 3.0

    for ri, version in enumerate(VERSIONS):
        for ci, size_mm in enumerate(SIZES_MM):
            x = margin_left + ci * (cell_w + col_gap)
            y = margin_top + ri * (cell_w + row_gap)
            qr_size_mm = min(size_mm, cell_w - 5)
            px = int(round(qr_size_mm / MM_PER_INCH * DPI))
            N = 17 + 4 * version
            ppm = int(round(px / N))

            patch, _ = generate_qr_patch(
                version=version, content=URL, ecl_str="L",
                ppm=ppm, quiet_zone_modules=4,
            )
            tmp_path = OUT_DIR / f"_tmp_qr_v{version}_s{size_mm}.png"
            cv2.imwrite(str(tmp_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            pdf.image(str(tmp_path), x=x, y=y, w=qr_size_mm, h=qr_size_mm)

            pdf.set_font("Helvetica", size=7)
            pdf.text(x=x, y=y + qr_size_mm + 2, text=f"V{version} {size_mm}mm")

    pdf.set_font("Helvetica", "B", size=14)
    pdf.text(x=margin_left, y=10, text=f"QR Test Dataset - {URL}")
    pdf.set_font("Helvetica", size=9)
    pdf.text(x=margin_left, y=15, text="Print at 300 DPI, laminate, cut out")

    out_path = OUT_DIR / "qr_sheet.pdf"
    pdf.output(str(out_path))

    for tmp in OUT_DIR.glob("_tmp_qr_*.png"):
        tmp.unlink()

    print(f"Saved {out_path}")
    print(f"  {len(VERSIONS)} versions x {len(SIZES_MM)} sizes = {len(VERSIONS)*len(SIZES_MM)} QR codes")
    for v in VERSIONS:
        for s in SIZES_MM:
            print(f"    V={v:2d}  {s:3d}mm")


if __name__ == "__main__":
    main()
