#!/usr/bin/env python3
"""Draw every paper figure and table from a data directory.

    python figures/make_all.py                       # paper_data/ -> build/
    python figures/make_all.py --data results/paper  # fresh re-run export
    python figures/make_all.py --only exp2

Figures are written to <out>/figures/*.pdf and table bodies to <out>/tables/*.tex.
The two 3D geometry renderings are drawn by figures/waveguide_geometry.m (MATLAB).
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import appendixA  # noqa: E402
import exp1  # noqa: E402
import exp2  # noqa: E402
import exp3  # noqa: E402

PARTS = {'exp1': exp1, 'exp2': exp2, 'exp3': exp3, 'appendixA': appendixA}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=HERE.parent / 'paper_data', type=Path)
    ap.add_argument('--out', default=HERE.parent / 'build', type=Path)
    ap.add_argument('--only', choices=sorted(PARTS), action='append')
    args = ap.parse_args()
    figures, tables = args.out / 'figures', args.out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    for name in args.only or PARTS:
        part = PARTS[name]
        part.figures(args.data / name, figures)
        for stem, text in part.tables(args.data / name).items():
            (tables / f'{stem}.tex').write_text(text)
        print(f'{name}: done')


if __name__ == '__main__':
    main()
