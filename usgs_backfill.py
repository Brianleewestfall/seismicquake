#!/usr/bin/env python3
"""
USGS Historical Earthquake Backfill for TeslaQuake
===================================================
Downloads M4.0+ earthquakes from USGS ComCat API and inserts into
the historical_earthquakes table in Supabase.

Usage:
    python usgs_backfill.py --year 2021
    python usgs_backfill.py --start 2001 --end 2025
    python usgs_backfill.py --fill-gaps
    python usgs_backfill.py --year 2021 --dry-run

Environment:
    TESLAQUAKE_SUPABASE_URL  - Supabase project URL
    TESLAQUAKE_SUPABASE_KEY  - Supabase service role key
"""

import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
import argparse
from io import StringIO

USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_COUNT = "https://earthquake.usgs.gov/fdsnws/event/1/count"
MIN_MAGNITUDE = 4.0
BATCH_SIZE = 500
TABLE = "historical_earthquakes"

SUPABASE_URL = os.environ.get("TESLAQUAKE_SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("TESLAQUAKE_SUPABASE_KEY", "")

EXISTING_YEARS = [1932, 1936, 1939, 1996, 1997, 1998, 1999,
                  2000, 2003, 2005, 2010, 2015, 2020, 2024, 2026]
MISSING_YEARS = sorted(set(range(2000, 2026)) - set(EXISTING_YEARS))


def count_events(year):
    url = (f"{USGS_COUNT}?starttime={year}-01-01"
           f"&endtime={year}-12-31&minmagnitude={MIN_MAGNITUDE}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return int(resp.read().decode().strip())
    except Exception as e:
        print(f"  Warning: Count failed for {year}: {e}")
        return -1


def download_year(year):
    url = (f"{USGS_API}?format=csv&starttime={year}-01-01"
           f"&endtime={year}-12-31&minmagnitude={MIN_MAGNITUDE}"
           f"&orderby=time&limit=20000")
    print(f"  Downloading {year} from USGS...")
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "TeslaQuake-Backfill/1.0")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read().decode('utf-8')
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def parse_csv(csv_text):
    rows = []
    reader = csv.DictReader(StringIO(csv_text))
    for r in reader:
        try:
            ts = r.get('time', '')
            if not ts:
                continue
            mag = float(r.get('mag', 0) or 0)
            lat = float(r.get('latitude', 0) or 0)
            lon = float(r.get('longitude', 0) or 0)
            depth = float(r.get('depth', 0) or 0)
            place = (r.get('place', '') or '')[:500]
            usgs_id = r.get('id', '')
            mag_type = r.get('magType', '')
            date_str = ts[:10] if len(ts) >= 10 else ''
            region = ''
            if ',' in place:
                region = place.split(',')[-1].strip()[:100]
            rows.append({
                "usgs_id": usgs_id,
                "timestamp": ts,
                "date": date_str,
                "latitude": lat,
                "longitude": lon,
                "depth_km": depth,
                "magnitude": mag,
                "magnitude_type": mag_type,
                "place": place if place else None,
                "region": region if region else None,
            })
        except (ValueError, KeyError):
            continue
    return rows


def post_batch(rows):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "NO_CREDENTIALS"
    url = f"{SUPABASE_URL}/rest/v1/{TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal,resolution=merge-duplicates",
    }
    data = json.dumps(rows).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8')[:200]
        return f"HTTP {e.code}: {body}"
    except Exception as e:
        return f"ERROR: {e}"


def insert_rows(rows, dry_run=False):
    total = len(rows)
    inserted = 0
    errors = 0
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        if dry_run:
            inserted += len(batch)
            continue
        status = post_batch(batch)
        if status in (200, 201):
            inserted += len(batch)
        else:
            errors += 1
            print(f"    Batch error: {status}")
            time.sleep(2)
            status2 = post_batch(batch)
            if status2 in (200, 201):
                inserted += len(batch)
                errors -= 1
        if not dry_run:
            time.sleep(0.5)
    return {"total": total, "inserted": inserted, "errors": errors}


def backfill_year(year, dry_run=False):
    print(f"\n{'='*50}")
    print(f"  Year {year}")
    print(f"{'='*50}")
    count = count_events(year)
    if count < 0:
        return None
    print(f"  USGS reports {count} M{MIN_MAGNITUDE}+ events")
    csv_text = download_year(year)
    if not csv_text:
        return None
    rows = parse_csv(csv_text)
    print(f"  Parsed {len(rows)} events")
    if not rows:
        return {"year": year, "total": 0, "inserted": 0, "errors": 0}
    m5 = sum(1 for r in rows if r['magnitude'] >= 5.0)
    m6 = sum(1 for r in rows if r['magnitude'] >= 6.0)
    m7 = sum(1 for r in rows if r['magnitude'] >= 7.0)
    print(f"  M5+: {m5} | M6+: {m6} | M7+: {m7}")
    if dry_run:
        print(f"  DRY RUN - not inserting")
    else:
        batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Inserting {len(rows)} rows in {batches} batches...")
    result = insert_rows(rows, dry_run=dry_run)
    result["year"] = year
    ok = "OK" if result["errors"] == 0 else "ERRORS"
    print(f"  {ok}: {result['inserted']}/{result['total']} inserted")
    return result


def main():
    parser = argparse.ArgumentParser(description="USGS Earthquake Backfill")
    parser.add_argument("--year", type=int, help="Single year")
    parser.add_argument("--start", type=int, help="Start year")
    parser.add_argument("--end", type=int, help="End year")
    parser.add_argument("--fill-gaps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-gaps", action="store_true")
    args = parser.parse_args()

    if args.list_gaps:
        print(f"Missing years: {MISSING_YEARS}")
        for y in MISSING_YEARS:
            c = count_events(y)
            print(f"  {y}: {c:,} events")
        return

    if args.year:
        years = [args.year]
    elif args.start and args.end:
        years = list(range(args.start, args.end + 1))
    elif args.fill_gaps:
        years = MISSING_YEARS
    else:
        parser.print_help()
        return

    if not args.dry_run and (not SUPABASE_URL or not SUPABASE_KEY):
        print("Set TESLAQUAKE_SUPABASE_URL and TESLAQUAKE_SUPABASE_KEY")
        sys.exit(1)

    print(f"TeslaQuake USGS Earthquake Backfill")
    print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")
    results = []
    total_inserted = 0
    start_time = time.time()

    for year in years:
        result = backfill_year(year, dry_run=args.dry_run)
        if result:
            results.append(result)
            total_inserted += result["inserted"]
        time.sleep(1)

    elapsed = time.time() - start_time
    print(f"\nBACKFILL COMPLETE")
    print(f"Years: {len(results)} | Inserted: {total_inserted:,} | Time: {elapsed:.0f}s")
    for r in results:
        print(f"  {r['year']}: {r['inserted']:,}/{r['total']:,}")


if __name__ == "__main__":
    main()
