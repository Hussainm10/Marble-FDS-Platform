"""Risk Score Decay Cron Service.

Daily job that reduces risk_score for entities with extended clean periods.
Connects directly to Checkmarble's PostgreSQL to find records, then
re-ingests updated scores via the Checkmarble HTTP API.

Decay rules (spec 5.3):
    7+  clean days  ->  -5 points
    14+ clean days  ->  -10 points
    30+ clean days  ->  reset to 0

"Clean days" = days since last activity:
    individual_users  ->  last_transaction_date
    agents            ->  last_login
    merchant_users    ->  last_activity
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone

import httpx
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PG_HOST = os.getenv("PG_HOSTNAME", "db")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DATABASE = os.getenv("PG_DATABASE", "marble")

CHECKMARBLE_API_URL = os.getenv("CHECKMARBLE_API_URL", "http://api:8080").rstrip("/")
CHECKMARBLE_API_KEY = os.getenv("CHECKMARBLE_API_KEY", "")

DECAY_INTERVAL_HOURS = int(os.getenv("DECAY_INTERVAL_HOURS", "24"))
STARTUP_DELAY_SECONDS = int(os.getenv("STARTUP_DELAY_SECONDS", "30"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

# Table configs: (checkmarble_table_name, id_field, last_activity_field)
TABLE_CONFIGS = [
    ("individual_users", "user_id", "last_transaction_date"),
    ("agents", "agent_id", "last_login"),
    ("merchant_users", "merchant_uid", "last_activity"),
]

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("risk_decay")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_db_connection():
    """Open a connection to Checkmarble's PostgreSQL."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DATABASE,
    )


def discover_table_locations(conn):
    """Find schemas where each target table lives with a risk_score column.

    Returns dict: { table_name: schema_name }
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_schema, table_name
            FROM information_schema.columns
            WHERE column_name = 'risk_score'
              AND table_schema NOT IN ('information_schema', 'pg_catalog')
        """)
        rows = cur.fetchall()

    locations = {}
    for schema, table in rows:
        locations[table] = schema
    return locations


def verify_column_exists(conn, schema, table, column):
    """Check that a column exists in the given schema.table."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """, (schema, table, column))
        return cur.fetchone() is not None


def get_data_columns(conn, schema, table):
    """Get all user-defined columns (excluding Checkmarble internal columns)."""
    internal = {"id", "object_id", "updated_at", "valid_from", "valid_until"}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (schema, table))
        return [r[0] for r in cur.fetchall() if r[0] not in internal]


def fetch_records_needing_decay(conn, schema, table, id_field, activity_field):
    """Return current-version records where risk_score > 0.

    Checkmarble uses temporal tables: only rows with valid_until = 'infinity'
    are the current version.
    """
    query = (
        f'SELECT "object_id", "{id_field}", "risk_score", "{activity_field}" '
        f'FROM "{schema}"."{table}" '
        f"WHERE \"risk_score\" > 0 AND \"valid_until\" = 'infinity'"
    )
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        return cur.fetchall()


def fetch_full_record(conn, schema, table, object_id, columns):
    """Read all data columns for a record (current version only)."""
    cols = ", ".join(f'"{c}"' for c in columns)
    query = (
        f'SELECT "object_id", {cols} '
        f'FROM "{schema}"."{table}" '
        f"WHERE \"object_id\" = %s AND \"valid_until\" = 'infinity'"
    )
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (object_id,))
        return cur.fetchone()


# ---------------------------------------------------------------------------
# Decay logic
# ---------------------------------------------------------------------------


def calculate_new_score(current_score: float, last_activity, now: datetime) -> float:
    """Apply the decay formula.  Returns the new score (unchanged if < 7 days)."""
    if last_activity is None:
        return current_score  # Cannot determine clean period

    # Ensure timezone-aware comparison
    if last_activity.tzinfo is None:
        last_activity = last_activity.replace(tzinfo=timezone.utc)

    clean_days = (now - last_activity).days

    if clean_days >= 30:
        return 0
    elif clean_days >= 14:
        return max(0, current_score - 10)
    elif clean_days >= 7:
        return max(0, current_score - 5)
    else:
        return current_score


# ---------------------------------------------------------------------------
# Checkmarble re-ingestion
# ---------------------------------------------------------------------------


def _serialize_value(val):
    """Convert a Python value to JSON-safe format for Checkmarble ingestion."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, (int, float, bool, str)):
        return val
    return str(val)


def reingest_record(table_name: str, record: dict):
    """POST the full record to Checkmarble ingestion API.

    Checkmarble creates a new temporal version of the record; sending all
    fields ensures none are lost in the new version.
    """
    url = f"{CHECKMARBLE_API_URL}/ingestion/{table_name}"
    payload = {"updated_at": datetime.now(timezone.utc).isoformat()}

    for key, val in record.items():
        serialized = _serialize_value(val)
        if serialized is not None:
            payload[key] = serialized

    headers = {
        "X-API-Key": CHECKMARBLE_API_KEY,
        "Content-Type": "application/json",
    }
    resp = httpx.post(url, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Main decay run
# ---------------------------------------------------------------------------


def run_decay():
    """Execute a single decay cycle across all configured tables."""
    now = datetime.now(timezone.utc)
    logger.info("=== Decay run started at %s ===", now.isoformat())

    conn = get_db_connection()
    try:
        locations = discover_table_locations(conn)
        if not locations:
            logger.warning("No tables with risk_score column found in database")
            return

        logger.info("Discovered tables: %s", {t: s for t, s in locations.items()})

        total_updated = 0
        total_scanned = 0

        for table_name, id_field, activity_field in TABLE_CONFIGS:
            schema = locations.get(table_name)
            if not schema:
                logger.warning("Table '%s' not found — skipping", table_name)
                continue

            # Verify the activity column exists
            if not verify_column_exists(conn, schema, table_name, activity_field):
                logger.warning(
                    "Column '%s' not found in %s.%s — skipping",
                    activity_field, schema, table_name,
                )
                continue

            # Get all data columns for full-record re-ingestion
            data_columns = get_data_columns(conn, schema, table_name)

            records = fetch_records_needing_decay(
                conn, schema, table_name, id_field, activity_field,
            )
            logger.info(
                "[%s] Found %d records with risk_score > 0",
                table_name, len(records),
            )
            total_scanned += len(records)

            updated = 0
            for rec in records:
                old_score = float(rec["risk_score"])
                last_activity = rec[activity_field]
                new_score = calculate_new_score(old_score, last_activity, now)

                if new_score == old_score:
                    continue

                # Calculate clean days for logging
                clean_days = 0
                if last_activity is not None:
                    act = last_activity
                    if act.tzinfo is None:
                        act = act.replace(tzinfo=timezone.utc)
                    clean_days = (now - act).days

                try:
                    # Read the full current record so re-ingestion preserves all fields
                    full_rec = fetch_full_record(
                        conn, schema, table_name, rec["object_id"], data_columns,
                    )
                    if full_rec is None:
                        logger.warning(
                            "  Record vanished: %s %s", table_name, rec[id_field],
                        )
                        continue

                    full_rec["risk_score"] = new_score
                    reingest_record(table_name, full_rec)
                    logger.info(
                        "  %s %s: %.0f -> %.0f (%d clean days)",
                        table_name, rec[id_field], old_score, new_score, clean_days,
                    )
                    updated += 1
                except Exception as e:
                    logger.error(
                        "  Failed to reingest %s %s: %s",
                        table_name, rec[id_field], e,
                    )

            total_updated += updated
            logger.info(
                "[%s] Updated %d / %d records", table_name, updated, len(records),
            )

        logger.info(
            "=== Decay run complete: %d updated, %d scanned ===",
            total_updated, total_scanned,
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point — sleep loop
# ---------------------------------------------------------------------------


def main():
    logger.info(
        "Risk Score Decay Service starting (interval=%dh, startup_delay=%ds)",
        DECAY_INTERVAL_HOURS, STARTUP_DELAY_SECONDS,
    )

    # Wait for dependent services to be ready
    if STARTUP_DELAY_SECONDS > 0:
        logger.info("Waiting %ds for services to initialize...", STARTUP_DELAY_SECONDS)
        time.sleep(STARTUP_DELAY_SECONDS)

    while True:
        try:
            run_decay()
        except Exception as e:
            logger.error("Decay run failed: %s", e, exc_info=True)

        logger.info("Sleeping %d hours until next run...", DECAY_INTERVAL_HOURS)
        time.sleep(DECAY_INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    main()
