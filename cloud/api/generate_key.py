#!/usr/bin/env python3
"""Generate an admin API key and insert it into Supabase.

Usage:
    DATABASE_URL=<supabase_postgres_url> python generate_key.py \
        --name "Board Admin" \
        --email "admin@loci.ai" \
        --label "board-admin-key"

Outputs the raw API key once. Store it securely — it cannot be recovered.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys

try:
    import psycopg2
except ImportError:
    sys.exit("Install psycopg2-binary: pip install psycopg2-binary")


def generate_raw_key() -> str:
    return "loci_" + secrets.token_hex(32)


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a LOCI admin API key")
    parser.add_argument("--name", required=True, help="Tenant display name")
    parser.add_argument("--email", required=True, help="Tenant email (unique)")
    parser.add_argument("--label", default="admin", help="Key label (default: admin)")
    parser.add_argument("--namespace", default="loci_admin", help="Qdrant collection prefix")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        sys.exit("ERROR: DATABASE_URL environment variable not set")

    raw_key = generate_raw_key()
    key_hash = hash_key(raw_key)
    prefix = raw_key[:12]  # "loci_" + first 7 chars

    conn = psycopg2.connect(database_url)
    try:
        with conn:
            with conn.cursor() as cur:
                # Insert or fetch tenant
                cur.execute(
                    """
                    INSERT INTO tenants (name, email, tier)
                    VALUES (%s, %s, 'pro')
                    ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name
                    RETURNING id
                    """,
                    (args.name, args.email),
                )
                tenant_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO api_keys (tenant_id, key_hash, prefix, namespace, label)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (tenant_id, key_hash, prefix, args.namespace, args.label),
                )
                key_id = cur.fetchone()[0]

        print(f"\n=== API Key Generated ===")
        print(f"Key ID   : {key_id}")
        print(f"Tenant   : {args.name} <{args.email}>")
        print(f"Label    : {args.label}")
        print(f"Prefix   : {prefix}")
        print(f"\nRAW KEY (save this — shown only once):")
        print(f"\n  {raw_key}\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
