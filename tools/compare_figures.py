#!/usr/bin/env python3
"""Compare figures in two directories pixel by pixel (PDFs are rasterized with pdftoppm).

    python tools/compare_figures.py build/figures tests/reference/figures
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def raster(path, dpi, workdir):
    path = Path(path)
    if path.suffix.lower() == '.pdf':
        stem = Path(workdir) / f'{path.parent.name}_{path.stem}'
        subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-singlefile', str(path), str(stem)], check=True)
        path = stem.with_suffix('.png')
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)


def compare(new_dir, ref_dir, dpi=100, names=None):
    """Return {name: max channel difference} (None if missing, -1 if the size differs)."""
    ref_dir, new_dir = Path(ref_dir), Path(new_dir)
    names = names or sorted(p.name for p in ref_dir.iterdir() if p.suffix in ('.pdf', '.png'))
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ('ref', 'new'):
            (Path(tmp) / sub).mkdir()
        for name in names:
            new = new_dir / name
            if not new.exists():
                result[name] = None
                continue
            a, b = raster(ref_dir / name, dpi, Path(tmp) / 'ref'), raster(new, dpi, Path(tmp) / 'new')
            result[name] = -1 if a.shape != b.shape else int(np.abs(a - b).max())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('new')
    ap.add_argument('reference')
    ap.add_argument('--dpi', type=int, default=100)
    args = ap.parse_args()
    result = compare(args.new, args.reference, args.dpi)
    width = max(map(len, result))
    for name, diff in result.items():
        status = 'missing' if diff is None else 'size differs' if diff < 0 else 'identical' if diff == 0 \
            else f'max pixel difference {diff}'
        print(f'{name:<{width}}  {status}')
    sys.exit(0 if all(d == 0 for d in result.values()) else 1)


if __name__ == '__main__':
    main()
