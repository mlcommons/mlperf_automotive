# Wiring the Hardened checker into `submission_checker.py`

This package adds the **Hardened Category** disclosure check on top of the existing
MLPerf Automotive submission checker. It is intentionally additive: no existing
behaviour changes, and the new check only fires when a system declares
`"status": "hardened"`.

## Why this slots in cleanly

The base checker already has everything the hook needs:

| Fact in `submission_checker.py` (commit `5d11f17`) | Consequence |
|---|---|
| `VALID_AVAILABILITIES` already contains `"hardened"` (line ~185) | The category *is* the system's `status`; no new enum needed. |
| `status` is read at line ~1808 as `available = system_json.get("status").lower()` | That is the exact point to branch on `hardened`. |
| `system_id_json = division/submitter/systems/<system_desc>.json` (line ~1795) | The hardened package lives right beside it as `<system_desc>_hardened.json`. |
| Checker is pure stdlib, returns booleans, logs `error`/`warning` | `hardened_checker` follows the same contract — no new dependency. |
| Benchmark models live in `MODEL_CONFIG[version]["models"]` | `hardened_checker` imports them rather than duplicating; falls back if run standalone. |

## The patch

**1. Import at the top of `submission_checker.py`:**

```python
from hardened_checker import check_hardened_package
```

**2. In `check_results_dir`, right after the `status` validation block
(immediately after the `if available not in VALID_AVAILABILITIES:` check,
~line 1814), add:**

```python
                    # Hardened category requires an additional disclosure package.
                    if available == "hardened":
                        hardened_json = os.path.join(
                            division, submitter, "systems",
                            system_desc + "_hardened.json"
                        )
                        if not check_hardened_package(
                            name, hardened_json, system_desc_id=system_desc
                        ):
                            results[name] = None
                            continue
```

That is the entire integration. `name`, `division`, `submitter`, `system_desc`,
and `results` are all already in scope at that point in the loop.

## Submission directory layout

```
<division>/<submitter>/
├── systems/
│   ├── drive_orin_adas.json            # existing system description (status: "hardened")
│   └── drive_orin_adas_hardened.json   # NEW: hardened package (this schema)
├── results/<system_desc>/<model>/<scenario>/...
├── measurements/...
└── code/...
```

The `system_desc_id` field inside the hardened package must equal the system
description's basename (`drive_orin_adas`); the checker enforces this so a package
can't be silently attached to the wrong results line.

## Standalone use (no integration required)

```bash
# validate one package
python3 hardened_checker.py <division>/<submitter>/systems/<sys>_hardened.json \
    --system-desc-id <sys>

# run the test suite
python3 -m unittest test_hardened_checker -v
```

Exit code is `0` when valid, `1` otherwise — suitable for CI on the results repo.

## Error vs. warning policy

- **error** → the submission is rejected (missing required section/field, ADAS/AD
  without safety manual or tool qualification, quantizer with no calibration
  description, application/harness with no run command, target-dependent autotuning
  with no autotuning config, an audit identifier that pins nothing, malformed
  hash/checksum, an undisclosed local patch).
- **warning** → a reviewer should look, but it's valid (no component marked
  production-intended, runtime BOM missing firmware/OS/engine, no AEC-Q100 listed,
  a compilation recipe naming a non-benchmark model — left as a warning so Open
  division can use custom model names).

## Optional: JSON-Schema gate

`schema/hardened_package.schema.json` is the normative structural spec. The runtime
checker does **not** require `jsonschema` (matching the base checker), but the test
suite will additionally validate the examples against the schema if `jsonschema` is
installed. If MLCommons later wants a hard schema gate in CI, add `jsonschema` to the
tooling requirements and validate before calling `validate_hardened_package`.
```
