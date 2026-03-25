#!/usr/bin/env python3
"""DataCortex Adversarial Test Suite — Pre-Publish Gate 3.

Generates 25+ edge case NDJSON/JSON files and verifies byte-exact
roundtrip on both fast and balanced modes. Must pass 100% before
any crates.io publish.

Usage: python3 scripts/adversarial_test.py [--fast-only] [--verbose]
"""

import json, random, subprocess, os, sys, filecmp, tempfile, datetime, shutil

random.seed(42)
FAST_ONLY = "--fast-only" in sys.argv
VERBOSE = "--verbose" in sys.argv
MODES = ["fast"] if FAST_ONLY else ["fast", "balanced"]
DCX = os.environ.get("DATACORTEX_BIN", "datacortex")

tmpdir = tempfile.mkdtemp(prefix="dcx-adversarial-")
passed = 0
failed = 0
errors = []


def test(name, rows, timeout=30):
    global passed, failed
    fname = os.path.join(tmpdir, f"{name}.ndjson")
    with open(fname, "w") as f:
        f.write("\n".join(rows) + "\n")
    sz = os.path.getsize(fname)

    for mode in MODES:
        dcx_path = f"{fname}.{mode}.dcx"
        rt_path = f"{fname}.{mode}.rt"
        try:
            r = subprocess.run(
                [DCX, "compress", fname, dcx_path, "-m", mode],
                capture_output=True, text=True, timeout=timeout,
            )
            if r.returncode != 0:
                failed += 1
                err = f"{name} [{mode}] COMPRESS FAIL (code {r.returncode}): {r.stderr.strip()[:80]}"
                errors.append(err)
                if VERBOSE:
                    print(f"  FAIL  {name:<40} {mode:<8} {sz:>8}B  compress error")
                continue

            comp = os.path.getsize(dcx_path)
            r = subprocess.run(
                [DCX, "decompress", dcx_path, rt_path],
                capture_output=True, text=True, timeout=timeout,
            )
            if r.returncode != 0:
                failed += 1
                err = f"{name} [{mode}] DECOMPRESS FAIL: {r.stderr.strip()[:80]}"
                errors.append(err)
                if VERBOSE:
                    print(f"  FAIL  {name:<40} {mode:<8} {sz:>8}B -> {comp:>6}B  decompress error")
                continue

            if not os.path.exists(rt_path) or not filecmp.cmp(fname, rt_path):
                failed += 1
                err = f"{name} [{mode}] DIFF MISMATCH"
                errors.append(err)
                if VERBOSE:
                    print(f"  FAIL  {name:<40} {mode:<8} {sz:>8}B -> {comp:>6}B  diff mismatch")
                continue

            passed += 1
            if VERBOSE:
                ratio = sz / comp if comp > 0 else 0
                print(f"  PASS  {name:<40} {mode:<8} {sz:>8}B -> {comp:>6}B  ({ratio:.1f}x)")

        except subprocess.TimeoutExpired:
            failed += 1
            errors.append(f"{name} [{mode}] TIMEOUT ({timeout}s)")
            if VERBOSE:
                print(f"  FAIL  {name:<40} {mode:<8} TIMEOUT")


# ==================== TEST CASES ====================

base = datetime.datetime(2026, 3, 25, 10, 0, 0)

# 1. Mixed types in same column (Bug #1)
test("mixed_types", [
    json.dumps({"val": [i, True, None, "str", 3.14, [1, 2]][i % 6], "idx": i}, separators=(",", ":"))
    for i in range(500)
])

# 2. Values near epoch range (Bug #2)
test("epoch_boundary", [
    json.dumps({"val": v, "idx": i}, separators=(",", ":"))
    for i, v in enumerate([946684800, 4102444800, 2147483647, -2147483648, 0, 1000000000, 2000000000] * 50)
])

# 3. Nested objects with varying sub-keys (Bug #3)
test("varying_nested", [
    json.dumps({"id": i, "meta": ({"x": i, "extra": i, "bonus": "yes"} if i % 3 == 0 else {"x": i} if i % 3 == 1 else {"x": i, "extra": i})}, separators=(",", ":"))
    for i in range(500)
])

# 4. Unquoted array values (Bug #4)
test("singleton_arrays", [
    json.dumps({"items": [{"x": i}], "id": i}, separators=(",", ":"))
    for i in range(500)
])

# 5. Very long strings (Bug #5)
test("long_strings", [
    json.dumps({"data": "A" * 100000 + f"_{i}", "id": i}, separators=(",", ":"))
    for i in range(50)
], timeout=60)

# 6. Identical rows (Bug #6)
test("identical_rows", [
    json.dumps({"x": 1}, separators=(",", ":")) for _ in range(10000)
])

# 7. Unicode heavy
test("unicode", [
    json.dumps({"name": random.choice(["hello", "caf\u00e9", "\u4e16\u754c", "\u2764\ufe0f"]), "id": i}, separators=(",", ":"))
    for i in range(500)
])

# 8. Escaped characters
test("escaped_chars", [
    json.dumps({"log": f"line {i}\ttab\nnewline", "q": f'He said "hi"', "bs": f"path\\dir{i}"}, separators=(",", ":"))
    for i in range(500)
])

# 9. Scientific notation floats
test("sci_floats", [
    json.dumps({"f": v, "i": i}, separators=(",", ":"))
    for i, v in enumerate([0.0, 1e-10, 1e10, 3.141592653589793, -0.0, 9.999e100] * 80)
])

# 10. Null heavy (80% null)
test("null_heavy", [
    json.dumps({f"c{j}": (None if random.random() < 0.8 else random.randint(0, 100)) for j in range(20)}, separators=(",", ":"))
    for _ in range(1000)
])

# 11. Single column
test("single_column", [
    json.dumps({"value": i}, separators=(",", ":")) for i in range(5000)
])

# 12. Mixed schemas (3 variants)
test("mixed_schemas", [
    json.dumps(
        {"type": "a", "x": i, "y": "hello"} if i % 3 == 0
        else {"type": "b", "count": i, "active": True} if i % 3 == 1
        else {"type": "c", "data": [i, i + 1], "meta": {"k": i}},
        separators=(",", ":"),
    )
    for i in range(900)
])

# 13. Empty and singleton arrays
test("empty_arrays", [
    json.dumps({"data": [], "tags": [], "id": i}, separators=(",", ":"))
    for i in range(500)
])

# 14. Explicit null vs absent key
test("null_vs_absent", [
    json.dumps({"id": i, "meta": {"x": i, "y": None}} if i % 2 == 0 else {"id": i, "meta": {"x": i}}, separators=(",", ":"))
    for i in range(500)
])

# 15. Deep nesting (5 levels)
test("deep_nesting", [
    json.dumps({"a": {"b": {"c": {"d": {"e": i}}}}, "id": i}, separators=(",", ":"))
    for i in range(200)
])

# 16. Boolean vs string boolean
test("bool_vs_string", [
    json.dumps({"active": True if i % 2 == 0 else "true", "id": i}, separators=(",", ":"))
    for i in range(500)
])

# 17. Integer vs string integer
test("int_vs_string", [
    json.dumps({"count": 42 if i % 2 == 0 else "42", "id": i}, separators=(",", ":"))
    for i in range(500)
])

# 18. Many columns (100)
test("many_columns", [
    json.dumps({f"col_{j}": j + i for j in range(100)}, separators=(",", ":"))
    for i in range(200)
])

# 19. Boolean heavy (10 cols)
test("boolean_heavy", [
    json.dumps({f"b{j}": random.choice([True, False]) for j in range(10)}, separators=(",", ":"))
    for _ in range(5000)
])

# 20. Timestamp heavy (multiple formats)
test("timestamp_heavy", [
    json.dumps({
        "ts1": (base + datetime.timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "ts2": (base + datetime.timedelta(seconds=i * 2)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00",
        "id": i,
    }, separators=(",", ":"))
    for i in range(5000)
])

# 21. Production: Stripe webhooks
test("stripe_webhooks", [
    json.dumps({
        "type": random.choice(["charge.succeeded", "payment_intent.created", "refund.created"]),
        "data": {"object": {"id": f"ch_{random.randint(0, 0xFFFFFF):06x}", "amount": random.randint(100, 99999),
                            **({"card": {"brand": "visa", "last4": f"{random.randint(1000, 9999)}"}} if i % 3 == 0 else {})}},
        "created": int((base + datetime.timedelta(seconds=i)).timestamp()),
        "livemode": random.choice([True, False]),
    }, separators=(",", ":"))
    for i in range(1000)
])

# 22. Production: Docker logs
test("docker_logs", [
    json.dumps({
        "log": json.dumps({"level": "INFO", "msg": f"Request {i}", "dur": random.randint(1, 500)}),
        "stream": random.choice(["stdout", "stderr"]),
        "time": f"2026-03-25T{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}.{random.randint(0, 999):03d}Z",
    }, separators=(",", ":"))
    for i in range(2000)
])

# 23. Production: k8s events
test("k8s_events", [
    json.dumps({
        "ts": (base + datetime.timedelta(milliseconds=i * 100)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": random.choice(["INFO", "INFO", "INFO", "WARN", "ERROR"]),
        "component": random.choice(["api-server", "scheduler", "controller"]),
        "msg": random.choice(["Pod scheduled", "Health check OK", "Timeout", "OOM killed", "Scaling up"]),
        "ns": random.choice(["default", "kube-system", "monitoring"]),
    }, separators=(",", ":"))
    for i in range(5000)
])

# 24. Huge number of unique strings (high cardinality)
test("high_cardinality", [
    json.dumps({"session": f"sess_{random.randint(0, 0xFFFFFFFF):08x}", "id": i}, separators=(",", ":"))
    for i in range(2000)
])

# 25. Mixed array and object values
test("mixed_array_object", [
    json.dumps({"val": {"nested": i} if i % 3 == 0 else [i, i + 1] if i % 3 == 1 else i, "id": i}, separators=(",", ":"))
    for i in range(500)
])

# ==================== REPORT ====================

total = passed + failed
print(f"\n{'=' * 60}")
print(f"  DataCortex Adversarial Test Results")
print(f"  Binary: {DCX}")
print(f"  Modes: {', '.join(MODES)}")
print(f"{'=' * 60}")
print(f"  PASSED: {passed}/{total}")
print(f"  FAILED: {failed}/{total}")

if errors:
    print(f"\n  Failures:")
    for e in errors:
        print(f"    - {e}")

print(f"{'=' * 60}")

# Cleanup
shutil.rmtree(tmpdir)

sys.exit(0 if failed == 0 else 1)
