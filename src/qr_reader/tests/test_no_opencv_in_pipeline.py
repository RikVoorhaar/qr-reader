"""The main pipeline must not depend on OpenCV.

OpenCV is allowed for synthetic image generation (`qr_gen`, `synth`) and for
verification tooling (`scripts/`, `tests/`).  Nothing reachable from the
production entry points may import it — or import a module that does.

Each check runs in a fresh subprocess, so modules imported by other tests in
the same pytest session cannot mask a regression.
"""

from __future__ import annotations

import subprocess
import sys

# Imported eagerly by cv2 users; pulled in indirectly by qr_gen and synth.
_BANNED_MODULES = ("cv2", "qr_reader.qr_gen", "qr_reader.synth")


def _assert_import_is_clean(entrypoint: str) -> None:
    code = (
        "import sys\n"
        f"import {entrypoint}\n"
        f"banned = [m for m in {_BANNED_MODULES!r} if m in sys.modules]\n"
        "if banned:\n"
        "    raise SystemExit('imported: ' + ', '.join(sorted(banned)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert completed.returncode == 0, (
        f"`{entrypoint}` pulled in modules that must stay out of the main "
        f"pipeline ({completed.stderr.strip()}).\n"
        "OpenCV belongs in testing/synthesis/verification code only; put "
        "shared image operations in `qr_reader.preprocess`."
    )


def test_pipeline_import_does_not_pull_in_opencv():
    _assert_import_is_clean("qr_reader.pipeline")


def test_decoder_import_does_not_pull_in_opencv():
    _assert_import_is_clean("qr_reader.decoder.decoder")
