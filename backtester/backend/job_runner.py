
import json
import os
import sys

import _paths  # noqa: F401


def main():
    job_dir = os.path.abspath(sys.argv[1])
    with open(os.path.join(job_dir, 'job.json'), encoding='utf-8') as f:
        job = json.load(f)

    import run_backtest as RB
    RB.REPORTS = job_dir
    RB.STATUS_PATH = os.path.join(job_dir, 'status.json')
    RB.RESULTS_PATH = os.path.join(job_dir, 'results.json')
    RB.CHARTDATA_DIR = os.path.join(job_dir, 'chartdata')

    RB.run(strategy=job.get('strategy'), params=job.get('params'),
           days=job.get('days'), timeframe=job.get('timeframe'),
           name=job.get('name'))

    # exit code mirrors the final state so the parent can catch hard crashes
    try:
        with open(RB.STATUS_PATH, encoding='utf-8') as f:
            ok = json.load(f).get('state') == 'done'
    except Exception:
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
