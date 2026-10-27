"""``loci`` command-line interface.

- ``loci migrate-layout`` — one-shot migration of a legacy per-epoch Qdrant
  deployment (``{prefix}loci_{epoch}`` / ``{prefix}loci_sum_{coarse}``
  collections) to the bounded two-collection layout
  (``{prefix}loci_data`` / ``{prefix}loci_summary``).

Usage:
    loci migrate-layout --qdrant-url http://localhost:6333 [--prefix t1_]
                        [--dry-run] [--delete-old]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# `loci migrate-layout` — legacy per-epoch collections → bounded layout
# ---------------------------------------------------------------------------

# Payload indexes the new clients create on the bounded collections.  The
# data collection carries Hilbert bucket indexes; the summary collection
# does not (summaries have no Hilbert payload — spatial queries reach them
# through the exact x/y/z Range conditions and the post-filter).  Both
# collections get FLOAT indexes on the exact-range fields so those Range
# conditions are cheap (mirrors ``_EXACT_RANGE_FIELDS`` in the clients).
_HILBERT_INDEX_FIELDS = ("hilbert_r4", "hilbert_r8", "hilbert_r12")
_INTEGER_INDEX_FIELDS = ("timestamp_ms",)
_KEYWORD_INDEX_FIELDS = ("scene_id", "scale_level")
_FLOAT_INDEX_FIELDS = ("x", "y", "z", "confidence")

_DATA_SUFFIX = "loci_data"
_SUMMARY_SUFFIX = "loci_summary"


def _qdrant_client(url: str) -> Any:
    """Build a Qdrant client (separate function so tests can stub it)."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=60)


def _legacy_collections(
    names: list[str], prefix: str
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split *names* into legacy raw / summary collections under *prefix*.

    Matches only the OLD per-epoch layout: ``{prefix}loci_{epoch}`` raw
    collections and ``{prefix}loci_sum_{coarse}`` summary collections.
    Both regexes are anchored and the prefix is escaped, so other tenants'
    collections and the new bounded collections never match.

    Returns:
        ``(raw, summaries)`` — lists of ``(collection_name, numeric_id)``
        sorted by numeric id.
    """
    raw_re = re.compile(rf"^{re.escape(prefix)}loci_(\d+)$")
    sum_re = re.compile(rf"^{re.escape(prefix)}loci_sum_(\d+)$")
    raw: list[tuple[str, int]] = []
    summaries: list[tuple[str, int]] = []
    for name in names:
        m = sum_re.match(name)
        if m:
            summaries.append((name, int(m.group(1))))
            continue
        m = raw_re.match(name)
        if m:
            raw.append((name, int(m.group(1))))
    raw.sort(key=lambda item: item[1])
    summaries.sort(key=lambda item: item[1])
    return raw, summaries


def _ensure_target_collection(
    client: Any, name: str, vectors_config: Any, *, hilbert: bool
) -> bool:
    """Create target collection *name* with the standard payload indexes.

    Returns True when the collection was created, False when it already
    existed.  An existing collection with a different vector size is fatal
    (points about to be copied verbatim would be rejected or corrupted).
    """
    if client.collection_exists(name):
        existing = client.get_collection(name).config.params.vectors
        existing_size = getattr(existing, "size", None)
        wanted_size = getattr(vectors_config, "size", None)
        if existing_size != wanted_size:
            sys.stderr.write(
                f"error: target collection {name!r} already exists with vector size "
                f"{existing_size}, but legacy collections use {wanted_size}\n"
            )
            sys.exit(1)
        return False

    from qdrant_client.models import PayloadSchemaType

    client.create_collection(collection_name=name, vectors_config=vectors_config)
    if hilbert:
        for field in _HILBERT_INDEX_FIELDS:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=PayloadSchemaType.INTEGER
            )
    for field in _INTEGER_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=PayloadSchemaType.INTEGER
        )
    for field in _KEYWORD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=PayloadSchemaType.KEYWORD
        )
    for field in _FLOAT_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=PayloadSchemaType.FLOAT
        )
    return True


def _copy_collection(
    client: Any, source: str, target: str, batch_size: int
) -> tuple[int, int, list[Any]]:
    """Copy every point of *source* into *target* verbatim.

    Points keep their ids, vectors, and payloads.  Ids already present in
    the target are skipped (and upserting the same id again would be a
    no-op rewrite of identical data anyway), so re-running is idempotent.

    Returns:
        ``(copied, skipped, source_ids)``.
    """
    from qdrant_client.models import PointStruct

    copied = 0
    skipped = 0
    source_ids: list[Any] = []
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        batch_ids = [p.id for p in points]
        source_ids.extend(batch_ids)
        existing = {
            r.id
            for r in client.retrieve(
                collection_name=target, ids=batch_ids, with_payload=False, with_vectors=False
            )
        }
        fresh = [p for p in points if p.id not in existing]
        skipped += len(points) - len(fresh)
        if fresh:
            client.upsert(
                collection_name=target,
                points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in fresh],
                wait=True,
            )
            copied += len(fresh)
        if offset is None:
            break
    return copied, skipped, source_ids


def _count_present(client: Any, target: str, ids: list[Any], batch_size: int) -> int:
    """Count how many of *ids* exist in *target* (batched retrieve)."""
    present = 0
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        present += len(
            client.retrieve(
                collection_name=target, ids=chunk, with_payload=False, with_vectors=False
            )
        )
    return present


def _cmd_migrate_layout(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        sys.stderr.write("error: --batch-size must be >= 1\n")
        sys.exit(2)
    if args.epoch_size_ms < 1:
        sys.stderr.write("error: --epoch-size-ms must be >= 1\n")
        sys.exit(2)

    client = _qdrant_client(args.qdrant_url)
    names = [c.name for c in client.get_collections().collections]
    raw, summaries = _legacy_collections(names, args.prefix)
    data_target = f"{args.prefix}{_DATA_SUFFIX}"
    summary_target = f"{args.prefix}{_SUMMARY_SUFFIX}"

    if not raw and not summaries:
        print(
            f"No legacy per-epoch collections found for prefix {args.prefix!r} — "
            "nothing to migrate."
        )
        return

    # Sanity check (informational): with the configured epoch size, the
    # newest legacy epoch should not start in the future.
    if raw:
        newest_start_ms = max(ep for _, ep in raw) * args.epoch_size_ms
        if newest_start_ms > int(time.time() * 1000) + 86_400_000:
            print(
                f"warning: newest legacy epoch starts at t={newest_start_ms}ms, which is "
                f"in the future — is --epoch-size-ms={args.epoch_size_ms} correct? "
                "(informational only; points are copied verbatim either way)"
            )

    plan = [(name, data_target) for name, _ in raw]
    plan += [(name, summary_target) for name, _ in summaries]
    counts = {name: client.count(collection_name=name, exact=True).count for name, _ in plan}
    total = sum(counts.values())

    print(f"Migration plan (prefix {args.prefix!r}):")
    for name, target in plan:
        print(f"  {name}  ->  {target}  ({counts[name]} points)")
    print(f"  total: {len(raw)} raw + {len(summaries)} summary collections, {total} points")

    if args.dry_run:
        print("Dry run — no changes made.")
        return

    # Vector size / distance sniffed from the first legacy collection; the
    # legacy layout used one config for every collection.
    vectors_config = client.get_collection(plan[0][0]).config.params.vectors
    if raw and _ensure_target_collection(client, data_target, vectors_config, hilbert=True):
        print(f"Created {data_target}")
    if summaries and _ensure_target_collection(
        client, summary_target, vectors_config, hilbert=False
    ):
        print(f"Created {summary_target}")

    total_copied = 0
    total_skipped = 0
    for name, target in plan:
        copied, skipped, source_ids = _copy_collection(client, name, target, args.batch_size)
        total_copied += copied
        total_skipped += skipped

        # Verified copy gate: every source point must exist in the target
        # (and the scroll must have seen every point the collection holds)
        # before the legacy collection may be dropped.
        source_count = client.count(collection_name=name, exact=True).count
        present = _count_present(client, target, source_ids, args.batch_size)
        if present != len(source_ids) or len(source_ids) != source_count:
            sys.stderr.write(
                f"error: verification failed for {name!r}: {source_count} points in "
                f"source, {len(source_ids)} scrolled, {present} present in {target!r}. "
                "Legacy collections were NOT deleted; re-run after fixing.\n"
            )
            sys.exit(1)

        line = (
            f"  {name} -> {target}: copied {copied}, skipped {skipped} "
            f"(already present), verified {present}/{source_count}"
        )
        if args.delete_old:
            client.delete_collection(name)
            line += " — deleted legacy collection"
        print(line)

    print(
        f"Done: {total_copied} points copied, {total_skipped} already present "
        f"({len(plan)} legacy collections"
        + (" deleted)." if args.delete_old else " kept — re-run with --delete-old to drop them).")
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loci", description="LOCI command-line interface")
    subs = parser.add_subparsers(dest="command", required=True)

    migrate = subs.add_parser(
        "migrate-layout",
        help="Migrate a legacy per-epoch Qdrant deployment to the bounded "
        "two-collection layout ({prefix}loci_data / {prefix}loci_summary)",
    )
    migrate.add_argument(
        "--qdrant-url", required=True, help="Qdrant URL, e.g. http://localhost:6333"
    )
    migrate.add_argument(
        "--prefix",
        default="",
        help="Tenant/collection prefix (default: none) — only collections under "
        "this prefix are migrated",
    )
    migrate.add_argument(
        "--epoch-size-ms",
        type=int,
        default=5000,
        help="Epoch width the deployment was created with (sanity checks only; default: 5000)",
    )
    migrate.add_argument(
        "--batch-size", type=int, default=256, help="Points per scroll/upsert batch (default: 256)"
    )
    migrate.add_argument(
        "--delete-old",
        action="store_true",
        help="Drop each legacy collection after its copy has been verified",
    )
    migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the migration plan (collections and point counts) without writing",
    )
    migrate.set_defaults(func=_cmd_migrate_layout)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
