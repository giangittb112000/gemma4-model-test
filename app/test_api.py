"""Smoke test: send sample queries to the running API and validate JSON schema."""

import json
import os
import sys
import time

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUIRED_KEYS = {"category", "product", "brand", "model", "attributes"}

SAMPLE_QUERIES = [
    "dt ip 256",
    "laptop dell i5 16gb",
    "ss s24 ultra 512",
    "tai nghe bluetooth",
    "ao thun nam mau den",
]


def wait_for_api(timeout: int = 1800) -> None:
    """Wait until /health reports the model is ready (first run downloads weights)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                print(f"[ok] API ready at {API_URL}")
                return
            print("[..] model still loading ...")
        except requests.RequestException:
            print(f"[..] waiting for API at {API_URL} ...")
        time.sleep(5)
    raise SystemExit(f"[fail] API not ready at {API_URL}")


def main() -> int:
    print(f"=== Query parser API smoke test ({API_URL}) ===")
    wait_for_api()

    failures = 0
    total_ms = 0.0
    for query in SAMPLE_QUERIES:
        start = time.time()
        try:
            resp = requests.post(f"{API_URL}/parse", json={"query": query}, timeout=180)
            resp.raise_for_status()
            parsed = resp.json()["parsed"]
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] '{query}': {exc}")
            failures += 1
            continue

        elapsed = (time.time() - start) * 1000
        total_ms += elapsed
        missing = REQUIRED_KEYS - parsed.keys()
        status = "PASS" if not missing else "FAIL"
        if missing:
            failures += 1
        print(f"[{status}] ({elapsed:6.0f} ms) '{query}'")
        print(f"        -> {json.dumps(parsed, ensure_ascii=False)}")
        if missing:
            print(f"        !! missing keys: {sorted(missing)}")

    n = len(SAMPLE_QUERIES)
    print(f"\n=== {n - failures}/{n} passed, avg {total_ms / max(n, 1):.0f} ms ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
