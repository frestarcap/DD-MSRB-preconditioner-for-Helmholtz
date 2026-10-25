#!/usr/bin/env python3
"""Verify six JCP generators reproduce current approved TeX byte for byte.

Outputs go to a temporary directory; manuscript files and figures are not rewritten.
"""
from pathlib import Path
import difflib
import hashlib
import json
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(target, expected):
    actual = target.read_text()
    approved = expected.read_text()
    if actual != approved:
        diff = ''.join(difflib.unified_diff(approved.splitlines(True), actual.splitlines(True),
                                          fromfile=str(expected), tofile='regenerated'))
        raise AssertionError(diff)
    print(f'PASS: {expected.name} (byte-identical)')
    assert target.read_bytes() == expected.read_bytes()


def main():
    with tempfile.TemporaryDirectory(prefix='jcp-tables-') as tmp:
        out = Path(tmp)
        half = runpy.run_path(str(ROOT / 'scripts/write_jcp_shifted_paper.py'))
        target = out / 'jcp_halfspace_timing.tex'
        half['write_halfspace_table'](target)
        check(target, ROOT / 'JCP' / target.name)
        reviewer = runpy.run_path(str(ROOT / 'scripts/write_jcp_reviewer_tables.py'))
        for name, text in reviewer['generate']().items():
            target = out / name; target.write_text(text)
            check(target, ROOT / 'JCP' / name)
        subprocess.run([sys.executable, str(ROOT / 'scripts/write_ecc_budget_paper.py'),
                        '--check-timing'], check=True, cwd=ROOT)
        wave = runpy.run_path(str(ROOT / 'scripts/write_jcp_waveguide_table.py'))
        target = out / 'jcp_waveguide_frequency.tex'
        wave['main'](table_target=target, table_only=True)
        check(target, ROOT / 'JCP' / target.name)
        validation = runpy.run_path(str(ROOT / 'scripts/write_jcp_3d_validation.py'))
        raw = validation['SOURCE'].read_bytes(); data = json.loads(raw)
        validation['validate'](data)
        target = out / 'jcp_waveguide_validation.tex'
        validation['write_table'](data, target, hashlib.sha256(raw).hexdigest())
        check(target, ROOT / 'JCP' / target.name)
    print('PASS: all six JCP tables reproduced without changing manuscript files')


if __name__ == '__main__':
    main()
