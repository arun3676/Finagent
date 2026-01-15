import argparse
import json
import pathlib
import sys
import urllib.request


def _load_env(env_path: pathlib.Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key] = val
    return env


def _post_json(url: str, api_key: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-check Qdrant contents (tickers present + optional required ticker)."
    )
    default_env_path = str(
        pathlib.Path(__file__).resolve().parents[1] / "backend" / ".env"
    )
    parser.add_argument(
        "--env",
        default=default_env_path,
        help="Path to backend .env (default: finagent/backend/.env next to this repo)",
    )
    parser.add_argument(
        "--require",
        metavar="TICKER",
        help="Fail if this ticker is not present in the collection",
    )
    args = parser.parse_args()

    env_path = pathlib.Path(args.env)
    env = _load_env(env_path)
    api_key = env.get("QDRANT_API_KEY")
    base_url = (env.get("QDRANT_URL") or "").rstrip("/")
    collection = env.get("QDRANT_COLLECTION_NAME") or "finagent_docs"

    if not api_key or not base_url:
        print("Missing QDRANT_URL or QDRANT_API_KEY in env.", file=sys.stderr)
        return 2

    tickers: set[str] = set()
    offset = None
    points_seen = 0
    while True:
        body: dict = {"limit": 512, "with_payload": ["ticker"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset

        data = _post_json(
            f"{base_url}/collections/{collection}/points/scroll", api_key, body
        )
        result = data.get("result") or {}
        points = result.get("points") or []
        points_seen += len(points)
        for p in points:
            t = (p.get("payload") or {}).get("ticker")
            if t:
                tickers.add(t)

        offset = result.get("next_page_offset")
        if not offset or not points:
            break

    counts: dict[str, int] = {}
    for t in sorted(tickers):
        count_data = _post_json(
            f"{base_url}/collections/{collection}/points/count",
            api_key,
            {
                "filter": {"must": [{"key": "ticker", "match": {"value": t}}]},
                "exact": True,
            },
        )
        counts[t] = (count_data.get("result") or {}).get("count", 0)

    out = {
        "collection": collection,
        "unique_tickers": sorted(tickers),
        "unique_ticker_count": len(tickers),
        "points_seen_via_scroll": points_seen,
        "counts": counts,
    }
    print(json.dumps(out, indent=2))

    if args.require and args.require.upper() not in tickers:
        print(f"ERROR: Missing required ticker in Qdrant: {args.require}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
