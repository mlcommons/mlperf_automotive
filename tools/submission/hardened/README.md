# MLPerf Automotive — Hardened Category submission package

Reference implementation of the **Hardened System** disclosure package and its
validator, turning the *Hardened Category Straw Man* into an executable spec that
plugs into the existing `tools/submission/submission_checker.py`.

A "Hardened" submission (`status: "hardened"` — already a valid availability in the
base checker) is a production-intended system. On top of the ordinary MLCommons
artifacts it must ship one extra file, `systems/<system_desc>_hardened.json`, that
discloses the production software path, a build-time/runtime software bill of
materials, OSS/proprietary/licensing provenance, a reproducible model-compilation
recipe, and profile-appropriate safety/security evidence.

Sensitive material is **not** placed in the file. Each component records an immutable
identifier (commit / build ID / checksum); the artifact itself is made available to a
neutral third-party auditor under the standard MLCommons audit + NDA process.

## Contents

| Path | What it is |
|---|---|
| `schema/hardened_package.schema.json` | Normative JSON Schema (draft-07) — the formal form of the straw man's "Solution space" fields. |
| `hardened_checker.py` | Pure-stdlib validator + CLI. Mirrors `submission_checker.py`'s error/warning idiom and enforces the cross-field rules a static schema can't. |
| `examples/drive_orin_adas_hardened.json` | Full, realistic ADAS package (BEVFormer + SSD, INT8, TensorRT-style autotuning). Doubles as the worked template. |
| `examples/ivi_qm_hardened.json` | Minimal IVI/QM package (Llama-3.1-8B voice assistant) — shows safety manual is *not* required for IVI. |
| `examples/invalid_adas_hardened.json` | Deliberately broken package used by tests to prove the checker catches real violations. |
| `test_hardened_checker.py` | 19 unit tests: valid examples, every cross-field rule, optional schema conformance. |
| `INTEGRATION.md` | Exact ~10-line patch to wire the check into `submission_checker.py`, plus directory layout. |

## Quick start

```bash
# validate a package (exit 0 = valid, 1 = invalid)
python3 hardened_checker.py examples/drive_orin_adas_hardened.json --system-desc-id drive_orin_adas

# run the tests
python3 -m unittest test_hardened_checker -v
```

## How the package maps to the straw man

| Straw man section | Schema property | Notable enforced rule |
|---|---|---|
| Production-intended software inventory | `production_intended_inventory[]` | each entry needs a pinning `identifier`; ≥1 should be production-intended (warning otherwise) |
| SBOM — build-time | `software_bill_of_materials.build_toolchain_bom[]` | `quantizer`/`calibration_tool` roles must describe the calibration set |
| SBOM — runtime | `software_bill_of_materials.runtime_execution_bom[]` | `application`/`benchmark_harness` must give the run command + LoadGen connection |
| OSS components | `oss_components[]` | a declared local patch must be referenced **or** described at feature level |
| Supplier/vendor components | `proprietary_components[]` | owner, build ID, interface role, unique checksum, external-availability flag all required |
| Licensing | `license_summary[]` | license family + binary/source redistribution booleans |
| Reproducibility | `reproducibility` | public repo refs (valid commit hash) + per-model recipe; **target-dependent autotuning requires `autotuning_config`** |
| Safety evidence | `safety_evidence` | **profile-driven:** IVI→security manual; ADAS/AD→security + safety manual + tool qualification |

## Design choices worth flagging to the WG

- **The category is the `status` field.** `"hardened"` already exists in the base
  checker's `VALID_AVAILABILITIES`, so no parallel taxonomy is introduced — the
  package simply attaches to a hardened system description.
- **Profiles (IVI / ADAS / AD) live in the package, not in `system_type`.** The base
  checker currently locks `system_type` to `"adas"`, so the three Hardened System
  Profiles from inference_rules 6.1 are carried in `hardened_profile` and drive which
  safety evidence is mandatory.
- **No new runtime dependency.** The validator is stdlib-only like the base checker;
  `jsonschema` is used only by the (skippable) schema-conformance test.
- **Errors reject; warnings inform.** This keeps the gate strict on
  reproducibility/provenance/safety while leaving room for Open-division and
  early-maturity submissions (e.g. a recipe naming a custom model is a warning).
```
