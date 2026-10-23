#!/usr/bin/env python3
"""Run an explicit argv and retain output, exit code and timing without a shell."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cwd", type=Path, default=ROOT)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        parser.error("provide a command after --")
    if not args.label.replace("-", "").replace("_", "").isalnum():
        parser.error("label must contain only letters, numbers, - or _")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = ROOT / "logs/commands" / f"{stamp}_{args.label}"
    record = dict(argv=argv, cwd=str(args.cwd.resolve()),
                  started_at=datetime.now(timezone.utc).isoformat(),
                  output_path=str(base.with_suffix(".log")), status="RUNNING")
    meta = base.with_suffix(".json")

    def save():
        temp = meta.with_suffix(".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temp.replace(meta)

    save()
    start = time.monotonic()
    with base.with_suffix(".log").open("w") as output:
        try:
            process = subprocess.Popen(argv, cwd=args.cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except OSError as exc:
            output.write(f"{type(exc).__name__}: {exc}\n")
            print(str(exc), file=sys.stderr)
            code = 127
    record.update(exit_code=code, elapsed_seconds=time.monotonic() - start,
                  ended_at=datetime.now(timezone.utc).isoformat(), status="FINISHED")
    save()
    print(f"[log] {meta}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
