"""Check that the trading system agrees with itself. Exit code 1 if anything fails.

    .venv\\Scripts\\python.exe scripts\\verify_consistency.py            one report
    .venv\\Scripts\\python.exe scripts\\verify_consistency.py --watch 30  re-run every 30s
    .venv\\Scripts\\python.exe scripts\\verify_consistency.py --json      machine-readable

Reads Redis only: the server-side checks plus the node's own latest self-check
report (account vs fills, cache integrity, net positions, stuck orders, ...).
The same report is on the dashboard's Health tab.
"""
import argparse
import json
import sys
import time
from datetime import datetime

from trading import consistency
from trading import redis_io as K

ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "skip"}


def run_once(as_json: bool) -> int:
    rep = consistency.server_checks(consistency.gather(K.connect(K.live_url()), K.connect(K.test_url())))
    if as_json:
        print(json.dumps(rep, indent=2))
    else:
        s = rep["summary"]
        print(f"\n{datetime.now():%Y-%m-%d %H:%M:%S}  overall {rep['status'].upper()}  "
              f"({s['pass']} pass, {s['warn']} warn, {s['fail']} fail, {s['skip']} skipped)")
        order = {"fail": 0, "warn": 1, "pass": 2, "skip": 3}
        for c in sorted(rep["checks"], key=lambda c: order[c["status"]]):
            print(f"  [{ICON[c['status']]}] {c['title']}: {c['detail']}")
            for item in c["items"][:10]:
                print(f"           - {item}")
            if len(c["items"]) > 10:
                print(f"           ... {len(c['items']) - 10} more")
    return 1 if rep["status"] == "fail" else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the raw report")
    ap.add_argument("--watch", type=float, metavar="SECONDS", help="repeat every N seconds")
    args = ap.parse_args()
    if not args.watch:
        sys.exit(run_once(args.json))
    try:
        while True:
            run_once(args.json)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
