"""Select SWE-bench Verified instances for evaluation arms.

Deterministic, reproducible selection per the round-1 protocol:

- Source manifest: SWE-bench Verified (500 instances), downloaded from the
  ModelScope mirror of the official HF dataset (huggingface.co is not
  reachable from every environment).
- One random stream seeded with 42:
    1. shuffle the repo order,
    2. shuffle each repo's instance ids,
    3. round-robin across repos (repo rotation),
    4. first 50 draws -> dev-50 (the future formal dev set, frozen here),
    5. next 10 draws  -> smoke-10 (round-0 pipeline smoke).
- dev-50 and smoke-10 are disjoint by construction (single rotation stream,
  smoke continues after dev-50), so smoke instances never contaminate the
  formal dev set.

Usage:
    uv run python scripts/select_swe_tasks.py \
        [--parquet path/to/test-00000-of-00001.parquet]

Outputs (relative to the repo root):
    evaluation/swe_bench/dev-50-swe.json
    evaluation/swe_bench/smoke-10.json

Requires: pyarrow (or pandas) to read the parquet manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

SEED = 42
DEV_COUNT = 50
SMOKE_COUNT = 10

MANIFEST_URL = (
    "https://modelscope.cn/api/v1/datasets/princeton-nlp/SWE-bench_Verified/"
    "repo?Revision=master&FilePath=data/test-00000-of-00001.parquet"
)

OUTPUTS_DIR = Path("evaluation/swe_bench")


def load_manifest(parquet_path: Path | None) -> list[dict[str, str]]:
    """Return [{"instance_id": ..., "repo": ...}, ...] sorted by instance_id."""
    path = parquet_path
    if path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS_DIR / "SWE-bench_Verified.parquet"
        if not path.exists():
            print(f"downloading manifest -> {path}")
            request = urllib.request.Request(MANIFEST_URL, headers={"User-Agent": "curl/8"})
            data = urllib.request.urlopen(request).read()
            path.write_bytes(data)
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["instance_id", "repo"])
        ids = table.column("instance_id").to_pylist()
        repos = table.column("repo").to_pylist()
    except ImportError:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(path, columns=["instance_id", "repo"])
        ids = frame["instance_id"].tolist()
        repos = frame["repo"].tolist()
    rows = [{"instance_id": iid, "repo": repo} for iid, repo in zip(ids, repos)]
    if len(rows) != 500:
        raise SystemExit(f"expected 500 Verified instances, got {len(rows)}")
    return sorted(rows, key=lambda row: row["instance_id"])


def rotate_stream(
    rows: list[dict[str, str]], rng: random.Random
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Draw dev-50 then smoke-10 from ONE rotation stream.

    A single queue is built (repos shuffled, ids shuffled) and 60 items are
    round-robined out of it: the first 50 form dev-50, the next 10 form
    smoke-10.  Disjointness holds because each draw removes its item from
    the shared queue — no pool is ever rebuilt.
    """
    by_repo: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_repo.setdefault(row["repo"], []).append(row)
    repos = sorted(by_repo)
    rng.shuffle(repos)
    for repo in repos:
        rng.shuffle(by_repo[repo])
    queue = [by_repo[repo] for repo in repos]

    picked: list[dict[str, str]] = []
    while len(picked) < DEV_COUNT + SMOKE_COUNT and queue:
        for bucket in list(queue):
            if len(picked) == DEV_COUNT + SMOKE_COUNT:
                break
            picked.append(bucket.pop(0))
            if not bucket:
                queue.remove(bucket)
    if len(picked) != DEV_COUNT + SMOKE_COUNT:
        raise SystemExit(f"rotation exhausted with only {len(picked)} instances")
    return picked[:DEV_COUNT], picked[DEV_COUNT:]


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=None,
                        help="cached Verified manifest parquet (default: download once)")
    args = parser.parse_args()

    rows = load_manifest(args.parquet)
    rng = random.Random(SEED)

    dev, smoke = rotate_stream(rows, rng)

    dev_ids = {row["instance_id"] for row in dev}
    smoke_ids = {row["instance_id"] for row in smoke}
    if dev_ids & smoke_ids:
        raise SystemExit("rotation stream overlapped dev-50 and smoke-10")

    per_repo: OrderedDict[str, int] = OrderedDict()
    for row in dev:
        per_repo[row["repo"]] = per_repo.get(row["repo"], 0) + 1

    provenance = {
        "source": MANIFEST_URL,
        "seed": SEED,
        "rule": "single rotation stream: shuffle repo order + per-repo ids "
                "(random.Random(42)), round-robin; first 50 = dev-50, next 10 = smoke-10",
        "manifest_rows": len(rows),
    }
    write_json(OUTPUTS_DIR / "dev-50-swe.json", {
        **provenance,
        "count": len(dev),
        "per_repo": per_repo,
        "instance_ids": [row["instance_id"] for row in dev],
    })
    write_json(OUTPUTS_DIR / "smoke-10.json", {
        **provenance,
        "count": len(smoke),
        "instance_ids": [row["instance_id"] for row in smoke],
    })

    print("\nsmoke-10:")
    for row in smoke:
        print(f"  {row['instance_id']}  ({row['repo']})")
    print(f"\ndev-50 per repo: {dict(per_repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
