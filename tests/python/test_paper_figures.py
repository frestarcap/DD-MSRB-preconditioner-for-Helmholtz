"""Redraw every Python figure and table from paper_data/ and compare with the published versions."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'tests/reference'
sys.path.insert(0, str(ROOT / 'tools'))
from compare_figures import compare  # noqa: E402


@pytest.fixture(scope='module')
def build(tmp_path_factory):
    out = tmp_path_factory.mktemp('build')
    subprocess.run([sys.executable, str(ROOT / 'figures/make_all.py'), '--out', str(out)], check=True)
    return out


@pytest.mark.skipif(shutil.which('pdftoppm') is None, reason='pdftoppm (poppler) is required')
def test_figures_identical(build):
    names = sorted(p.name for p in (REFERENCE / 'figures').glob('*.pdf'))
    assert len(names) == 25
    result = compare(build / 'figures', REFERENCE / 'figures', names=names)
    assert result == {name: 0 for name in names}


def test_tables_identical(build):
    for ref in sorted((REFERENCE / 'tables').glob('*.tex')):
        assert (build / 'tables' / ref.name).read_text() == ref.read_text(), ref.name
