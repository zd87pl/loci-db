#!/usr/bin/env python3
"""Copy the latest objects of a production space into a staging space, so boards tested
against staging have something to answer about ("where is my cup").

    python3 cloud/workers/scripts/seed_staging.py \
        --from-env ~/.loci/cloud.env --to-env ~/.loci/cloud-staging.env

Each env file is the one bootstrap.py prints for that deployment (URL, project, space and the
secret key, possibly inside a comment). It reads production's `digest` (the newest sighting per
room and label) and writes one observation per object to staging, as upserts with stable ids,
so running it again refreshes rather than duplicates. Nothing is written to production.
"""

from __future__ import annotations

import argparse
import re
import sys

from devices import call, read_env_file

MAX_BATCH = 50


def target(env_path: str) -> tuple[str, str, str]:
    env = read_env_file(env_path)
    url = env.get("LOCI_CLOUD_URL", "").rstrip("/")
    project, key = env.get("LOCI_CLOUD_PROJECT", ""), env.get("LOCI_CLOUD_KEY", "")
    if not url.startswith(("http://", "https://")) or not project or not key.startswith("loci_sk_"):
        sys.exit(
            f"{env_path}: needs LOCI_CLOUD_URL, LOCI_CLOUD_PROJECT and LOCI_CLOUD_KEY (loci_sk_...)"
        )
    return f"{url}/p/{project}/v1", key, env.get("LOCI_CLOUD_SPACE", "home")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from-env", required=True, help="production env file (read only)")
    ap.add_argument("--to-env", required=True, help="staging env file (written to)")
    a = ap.parse_args()
    src, src_key, src_space = target(a.from_env)
    dst, dst_key, dst_space = target(a.to_env)
    if src == dst and src_space == dst_space:
        sys.exit("source and target are the same space")

    objects = call(src, "GET", f"/spaces/{src_space}/digest?since_rev=0", src_key)["objects"]
    observations = []
    for o in objects:
        slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", f"{o['label']}_{o.get('room') or 'any'}")[:80]
        observations.append(
            {
                "id": f"seed_{slug}",
                "label": o["label"],
                "frame": o["frame"],
                "room": o.get("room"),
                "x": o["x"],
                "y": o["y"],
                "z": o["z"],
                "t": o["t"],
                "confidence": o["confidence"],
                "metadata": {"source": "seed_staging"},
            }
        )
    written = 0
    for i in range(0, len(observations), MAX_BATCH):
        batch = observations[i : i + MAX_BATCH]
        r = call(
            dst,
            "POST",
            f"/spaces/{dst_space}/observations",
            dst_key,
            {"observations": batch, "upsert": True},
        )
        written += int(r.get("accepted", 0)) + int(r.get("updated", 0))
    print(f"copied {len(observations)} objects from {src_space} to {dst_space} ({written} written)")


if __name__ == "__main__":
    main()
