# analysis/verify_entities.py
"""
Verify the ground-truth status of planted entities against live registries.

The dataset's deterministic judging only holds if our "fictional" entities really
are absent (and our control entities really are present). After arXiv publication
the fictional names become slopsquatting targets, so this must be re-run and
timestamped before every generation sweep (DATASET_V2_PLAN.md §2.4 / §5.2).

For each registry-checkable entity we compare its live existence to the seed's
`expect_exists`:
  * fake / typosquat / cross-ecosystem entities -> expect_exists == False
  * control (real) entities                     -> expect_exists == True
A mismatch is a CONFLICT (a seed whose ground truth has drifted) and must be fixed
before generation. Names we cannot check from here (Java/headers/flags/features/
algorithms/protocols/versions) are reported as MANUAL.

Stdlib-only (urllib); degrades gracefully offline (UNREACHABLE, not failure).

Usage:
  python -m analysis.verify_entities --seeds prompts/seeds_v2.jsonl \
      --out prompts/entity_verification.jsonl
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

UA = "code-hallucination-research (entity-verification)"
TIMEOUT = 8

# entity kinds we can check from a public registry, and how
REGISTRY_KINDS = {"py_module", "py_submodule", "npm", "crate", "go_pkg"}
MANUAL_KINDS = {
    "java_class", "c_header", "cpp_symbol", "build_flag", "lang_feature",
    "named_algo", "protocol", "version_feature", None,
}


def _http_status(url: str) -> int:
    """Return HTTP status code (200/404/...) or -1 if unreachable."""
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def _exists_pypi(name: str):
    code = _http_status(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    return None if code == -1 else (code == 200)


def _exists_npm(name: str):
    # scoped names (@scope/pkg) must have the slash encoded
    code = _http_status(f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='@')}")
    return None if code == -1 else (code == 200)


def _exists_crate(name: str):
    code = _http_status(f"https://crates.io/api/v1/crates/{urllib.parse.quote(name)}")
    return None if code == -1 else (code == 200)


def _exists_go(path: str):
    code = _http_status(f"https://pkg.go.dev/{path}")
    return None if code == -1 else (code == 200)


def _exists_py_submodule(dotted: str):
    """Use importlib only if the base package is installed; else unverifiable."""
    import importlib.util
    base = dotted.split(".")[0]
    if importlib.util.find_spec(base) is None:
        return None  # base not installed locally -> cannot verify submodule here
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ModuleNotFoundError, ImportError, ValueError):
        return False


def check(entity: str, kind: str):
    """Return (exists: bool|None, method: str)."""
    if not entity or " " in entity:  # commands like 'pip install x' -> manual
        return None, "manual"
    if kind == "py_module":
        return _exists_pypi(entity), "pypi"
    if kind == "py_submodule":
        # try PyPI for a top-level typo, else importlib for a real submodule
        if "." not in entity:
            return _exists_pypi(entity), "pypi"
        return _exists_py_submodule(entity), "importlib"
    if kind == "npm":
        return _exists_npm(entity), "npm"
    if kind == "crate":
        return _exists_crate(entity), "crates.io"
    if kind == "go_pkg":
        return _exists_go(entity), "pkg.go.dev"
    return None, "manual"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="prompts/seeds_v2.jsonl")
    ap.add_argument("--out", type=str, default="prompts/entity_verification.jsonl")
    ap.add_argument("--offline", action="store_true", help="skip network checks")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.seeds).read_text().splitlines() if l.strip()]
    # unique (entity, kind, expect_exists)
    seen = {}
    for r in rows:
        e, k = r.get("entity"), r.get("entity_kind")
        if e is None:
            continue
        seen[(e, k)] = r.get("expect_exists")

    ts = datetime.now(timezone.utc).isoformat()
    results = []
    stats = Counter()
    for (entity, kind), expect in sorted(seen.items()):
        if args.offline or kind in MANUAL_KINDS or kind not in REGISTRY_KINDS:
            exists, method = None, "manual"
        else:
            exists, method = check(entity, kind)

        if exists is None:
            status = "MANUAL" if method == "manual" else "UNREACHABLE"
        elif expect is None:
            status = "INFO"  # checked but seed had no expectation
        elif exists == expect:
            status = "OK"
        else:
            status = "CONFLICT"  # ground truth drifted — must fix before generation
        stats[status] += 1
        results.append({
            "entity": entity, "entity_kind": kind, "expect_exists": expect,
            "exists": exists, "method": method, "status": status, "checked_at": ts,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Checked {len(results)} unique entities @ {ts}")
    for k in ("OK", "CONFLICT", "UNREACHABLE", "MANUAL", "INFO"):
        if stats[k]:
            print(f"  {k:12} {stats[k]}")
    conflicts = [r for r in results if r["status"] == "CONFLICT"]
    if conflicts:
        print("\n!! CONFLICTS (fix before generating) !!")
        for r in conflicts:
            print(f"  {r['entity_kind']:14} {r['entity']:40} "
                  f"expected exists={r['expect_exists']} but live={r['exists']}")
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
