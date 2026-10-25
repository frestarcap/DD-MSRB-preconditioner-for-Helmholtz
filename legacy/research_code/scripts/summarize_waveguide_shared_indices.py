#!/usr/bin/env python3
"""Validate and summarize the shared-index deployment inspection cohort."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/waveguide_shared_indices_20260913"
OLD = ROOT / "outputs/waveguide_whos_20260913"


def main():
    historical = json.loads((OLD / "summary.json").read_text())
    assert historical["validated"]
    versions, jobs, nodes, rows = set(), set(), set(), []
    for w in (24, 36, 48, 60):
        result = json.loads((OUT / "results" / f"w{w}_shared.json").read_text())
        assert result["validated"]
        assert (OUT / "results" / f"w{w}_source_sha256.txt").read_text() == (
            OLD / "results" / f"w{w}_source_sha256.txt").read_text()
        methods = {r["method"]: r for r in result["rows"]}
        assert set(methods) == {"global", "dd"}
        previous = next(r for r in historical["rows"] if r["omega"] == w)
        for method, r in methods.items():
            assert r["omega"] == w and r["N"] == previous["N"]
            assert r["retained_values_equal"] and r["actions_bitwise_equal"]
            assert r["checked_inverse_actions"] == 3 * r["layers"] == 9
            assert r["old_MiB"] == previous[f"{method}_whos_MiB"]
            versions.add(r["matlab_version"])
            jobs.add(r["job_id"])
            nodes.add(r["node"])
        g, d = methods["global"], methods["dd"]
        assert g["old_whos_bytes"] == g["shared_whos_bytes"]
        # Exactly two redundant double index maps, plus reduced container bytes.
        assert d["removed_logical_bytes"] >= 2 * 8 * d["N"]
        overhead = d["shared_MiB"] - previous["dd_formula_MiB"]
        assert 0 <= overhead < 0.11
        rows.append(dict(omega=w, global_whos_MiB=g["shared_MiB"],
                         dd_old_whos_MiB=d["old_MiB"],
                         dd_shared_whos_MiB=d["shared_MiB"],
                         dd_removed_MiB=d["removed_MiB"],
                         dd_formula_MiB=previous["dd_formula_MiB"],
                         dd_shared_auxiliary_MiB=overhead,
                         shared_saving_pct=100*(1-d["shared_MiB"]/g["shared_MiB"])))
    assert versions == {historical["matlab_version"]}
    assert len(jobs) == len(nodes) == 1
    job = next(iter(jobs))
    accounting = (OUT / "accounting.txt").read_text().splitlines()
    assert any(line.startswith(f"{job}|COMPLETED|0:0|") for line in accounting)
    assert "WAVEGUIDE_SHARED_INDEX_COHORT_PASS" in (
        OUT / "logs" / f"helm3d_sharedidx_{job}.out").read_text()
    summary = dict(validated=True, matlab_version=next(iter(versions)),
                   job_id=job, node=next(iter(nodes)), rows=rows,
                   scope="Logical deployment-object whos; no physical RAM or online speed claim")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
