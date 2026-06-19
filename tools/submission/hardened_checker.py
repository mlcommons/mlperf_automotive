"""Validator for the MLPerf Automotive 'Hardened System' submission package.

This module checks the *additional* disclosure package required by the Hardened
submission category (inference_rules.adoc 6.1) on top of the ordinary MLCommons
submission artifacts that ``submission_checker.py`` already validates.

Design goals (kept deliberately aligned with the existing tools/submission code):
  * Pure standard library -- no ``jsonschema`` runtime dependency, exactly like
    ``submission_checker.py``. The JSON Schema in schema/hardened_package.schema.json
    is the normative spec; this file is the executable enforcement that also covers
    the cross-field rules draft-07 cannot express.
  * Same reporting idiom: ``log.error`` for things that invalidate a submission and
    ``log.warning`` for things a reviewer should look at. The boolean return mirrors
    ``check_system_desc_id`` so it drops straight into ``check_results_dir``.
  * Keyed by package version (``v0.5`` / ``v1.0``) the same way MODEL_CONFIG keys the
    base checker, so the set of benchmark models is taken from the base config rather
    than duplicated here.

Integration: call ``check_hardened_package`` right after the base checker confirms
``status == "hardened"`` (submission_checker.py ~line 1808). See INTEGRATION.md.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hardened")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Profiles defined by inference_rules.adoc 6.1. The required evidence differs per
# profile; this table is the machine-readable form of that section's table.
#   security_manual  -> always required
#   safety_manual    -> required for ADAS/AD (the '*' / "depending on safety concept")
#   tool_qualification -> required for ADAS/AD
HARDENED_PROFILES = {
    "IVI": {
        "requires_security_manual": True,
        "requires_safety_manual": False,
        "requires_tool_qualification": False,
        "typical_safety_goal": "QM",
    },
    "ADAS": {
        "requires_security_manual": True,
        "requires_safety_manual": True,
        "requires_tool_qualification": True,
        "typical_safety_goal": "QM..ASIL-B",
    },
    "AD": {
        "requires_security_manual": True,
        "requires_safety_manual": True,
        "requires_tool_qualification": True,
        "typical_safety_goal": "ASIL-B and above",
    },
}

VALID_PACKAGE_VERSIONS = ["v0.5", "v1.0"]

VALID_PROVENANCE = ["open-source", "proprietary", "mixed"]

VALID_BUILD_ROLES = [
    "training_framework", "export_tooling", "ml_compiler", "optimizer",
    "quantizer", "calibration_tool", "packaging_tool", "container_builder", "other",
]
# Build roles whose output depends on a calibration set must describe it.
BUILD_ROLES_NEEDING_CALIBRATION = ["quantizer", "calibration_tool"]

VALID_RUNTIME_TYPES = [
    "firmware", "kernel_driver", "userspace_driver", "operating_system",
    "ai_runtime", "inference_engine", "safety_runtime", "application",
    "benchmark_harness", "other",
]
# Runtime components that actually drive the run must disclose the run command.
RUNTIME_TYPES_NEEDING_RUN_COMMAND = ["application", "benchmark_harness"]

# An audit identifier must carry at least one of these uniquely-pinning forms.
AUDIT_ID_FORMS = [
    "repository_url", "commit_hash", "release_tag", "build_id",
    "binary_checksum", "internal_source_identifier",
]

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_CHECKSUM_RE = re.compile(r"^(sha256|sha512|sha1|md5):[0-9a-fA-F]{32,128}$")


# ---------------------------------------------------------------------------
# Small helpers (kept local so the module is import-free beyond stdlib)
# ---------------------------------------------------------------------------

def _require(obj, field, ctx, errors):
    """Field must be present and non-empty (matching the base checker's notion of
    a 'meaningful response')."""
    if field not in obj or obj[field] in (None, "", [], {}):
        errors.append("%s: field '%s' is missing or empty" % (ctx, field))
        return False
    return True


def _check_audit_identifier(ident, ctx, errors):
    if not isinstance(ident, dict) or not ident:
        errors.append("%s: identifier must be a non-empty object with at least one "
                      "of %s" % (ctx, AUDIT_ID_FORMS))
        return
    if not any(ident.get(f) for f in AUDIT_ID_FORMS):
        errors.append("%s: identifier carries none of %s -- an auditor cannot pin the "
                      "exact build" % (ctx, AUDIT_ID_FORMS))
    commit = ident.get("commit_hash")
    if commit and not _COMMIT_RE.match(commit):
        errors.append("%s: commit_hash '%s' is not a valid hex hash" % (ctx, commit))
    checksum = ident.get("binary_checksum")
    if checksum and not _CHECKSUM_RE.match(checksum):
        errors.append("%s: binary_checksum '%s' must look like 'sha256:<hex>'"
                      % (ctx, checksum))


# ---------------------------------------------------------------------------
# Section checks
# ---------------------------------------------------------------------------

def _check_inventory(pkg, errors, warnings):
    items = pkg.get("production_intended_inventory")
    if not isinstance(items, list) or not items:
        errors.append("production_intended_inventory: must be a non-empty array")
        return
    saw_production = False
    for i, c in enumerate(items):
        ctx = "production_intended_inventory[%d]" % i
        for f in ("component_name", "functional_role", "supplier_or_owner"):
            _require(c, f, ctx, errors)
        for f in ("used_during_benchmark", "intended_for_production"):
            if not isinstance(c.get(f), bool):
                errors.append("%s: field '%s' must be a boolean" % (ctx, f))
        prov = c.get("provenance")
        if prov not in VALID_PROVENANCE:
            errors.append("%s: provenance '%s' not in %s" % (ctx, prov, VALID_PROVENANCE))
        if "identifier" in c:
            _check_audit_identifier(c["identifier"], ctx, errors)
        else:
            errors.append("%s: field 'identifier' is missing" % ctx)
        if c.get("intended_for_production"):
            saw_production = True
    if not saw_production:
        warnings.append("production_intended_inventory: no component is marked "
                        "intended_for_production -- a hardened submission is expected "
                        "to declare a production path")


def _check_bom(pkg, errors, warnings):
    bom = pkg.get("software_bill_of_materials")
    if not isinstance(bom, dict):
        errors.append("software_bill_of_materials: must be an object with "
                      "build_toolchain_bom and runtime_execution_bom")
        return

    build = bom.get("build_toolchain_bom")
    if not isinstance(build, list) or not build:
        errors.append("software_bill_of_materials.build_toolchain_bom: non-empty array required")
    else:
        for i, c in enumerate(build):
            ctx = "build_toolchain_bom[%d]" % i
            _require(c, "component_name", ctx, errors)
            _require(c, "version", ctx, errors)
            role = c.get("role")
            if role not in VALID_BUILD_ROLES:
                errors.append("%s: role '%s' not in %s" % (ctx, role, VALID_BUILD_ROLES))
            if role in BUILD_ROLES_NEEDING_CALIBRATION and not c.get(
                    "calibration_dataset_description"):
                errors.append("%s: role '%s' requires calibration_dataset_description"
                              % (ctx, role))
            if "identifier" in c:
                _check_audit_identifier(c["identifier"], ctx, errors)

    runtime = bom.get("runtime_execution_bom")
    if not isinstance(runtime, list) or not runtime:
        errors.append("software_bill_of_materials.runtime_execution_bom: non-empty array required")
        return

    seen_types = set()
    for i, c in enumerate(runtime):
        ctx = "runtime_execution_bom[%d]" % i
        _require(c, "component_name", ctx, errors)
        _require(c, "version", ctx, errors)
        ctype = c.get("component_type")
        if ctype not in VALID_RUNTIME_TYPES:
            errors.append("%s: component_type '%s' not in %s"
                          % (ctx, ctype, VALID_RUNTIME_TYPES))
        else:
            seen_types.add(ctype)
        if ctype in RUNTIME_TYPES_NEEDING_RUN_COMMAND and not c.get("run_command"):
            errors.append("%s: component_type '%s' requires run_command (how the app "
                          "connects to the runtime and to LoadGen)" % (ctx, ctype))
        if "identifier" in c:
            _check_audit_identifier(c["identifier"], ctx, errors)

    # The runtime stack of a production-intended ECU is expected to disclose, at
    # minimum, firmware/OS and the engine that executes the artifact.
    for expected in ("firmware", "operating_system"):
        if expected not in seen_types:
            warnings.append("runtime_execution_bom: no component of type '%s' disclosed"
                            % expected)
    if not ({"ai_runtime", "inference_engine", "safety_runtime"} & seen_types):
        warnings.append("runtime_execution_bom: no ai_runtime/inference_engine/"
                        "safety_runtime disclosed -- something must execute the artifact")


def _check_oss(pkg, errors, warnings):
    items = pkg.get("oss_components", [])
    if not isinstance(items, list):
        errors.append("oss_components: must be an array")
        return
    for i, c in enumerate(items):
        ctx = "oss_components[%d]" % i
        for f in ("component_name", "upstream_url", "immutable_identifier", "role"):
            _require(c, f, ctx, errors)
        # If a fork/patch exists it must be either referenced or described at feature
        # level (the rules explicitly allow "describe instead of share").
        if c.get("has_local_patches") and not (
                c.get("patch_reference") or c.get("patch_feature_description")):
            errors.append("%s: has_local_patches is true but neither patch_reference "
                          "nor patch_feature_description is provided" % ctx)


def _check_proprietary(pkg, errors, warnings):
    items = pkg.get("proprietary_components", [])
    if not isinstance(items, list):
        errors.append("proprietary_components: must be an array")
        return
    for i, c in enumerate(items):
        ctx = "proprietary_components[%d]" % i
        for f in ("component_name", "owner", "version_or_build_id", "interface_role",
                  "internal_identifier_or_checksum", "distribution_model"):
            _require(c, f, ctx, errors)
        if not isinstance(c.get("available_to_external_customers"), bool):
            errors.append("%s: available_to_external_customers must be a boolean" % ctx)


def _check_licensing(pkg, errors, warnings):
    items = pkg.get("license_summary", [])
    if not isinstance(items, list):
        errors.append("license_summary: must be an array")
        return
    for i, c in enumerate(items):
        ctx = "license_summary[%d]" % i
        _require(c, "component_name", ctx, errors)
        _require(c, "license_family", ctx, errors)
        for f in ("binary_redistribution_allowed", "source_redistribution_allowed"):
            if not isinstance(c.get(f), bool):
                errors.append("%s: field '%s' must be a boolean" % (ctx, f))


def _check_reproducibility(pkg, version, errors, warnings):
    repro = pkg.get("reproducibility")
    if not isinstance(repro, dict):
        errors.append("reproducibility: object required")
        return

    ref = repro.get("mlperf_automotive_repo_reference")
    if not isinstance(ref, dict):
        errors.append("reproducibility.mlperf_automotive_repo_reference: object required")
    else:
        for f in ("implementation_repo_url", "results_repo_url", "commit_hash"):
            _require(ref, f, "mlperf_automotive_repo_reference", errors)
        commit = ref.get("commit_hash")
        if commit and not _COMMIT_RE.match(commit):
            errors.append("mlperf_automotive_repo_reference: commit_hash '%s' is not a "
                          "valid hex hash" % commit)

    recipes = repro.get("model_compilation_recipe")
    if not isinstance(recipes, list) or not recipes:
        errors.append("reproducibility.model_compilation_recipe: non-empty array required")
        return

    valid_models = _models_for_version(version)
    for i, r in enumerate(recipes):
        ctx = "model_compilation_recipe[%d]" % i
        model = r.get("model")
        if not model:
            errors.append("%s: field 'model' is missing" % ctx)
        elif valid_models is not None and model not in valid_models:
            # Closed/network only; open submissions may use custom names so this is a
            # warning to leave room for open-division recipes.
            warnings.append("%s: model '%s' is not a known benchmark model %s"
                            % (ctx, model, sorted(valid_models)))
        if not (isinstance(r.get("steps"), list) and r["steps"]):
            errors.append("%s: 'steps' must be a non-empty array" % ctx)
        _require(r, "compiler_config", ctx, errors)
        # Target-dependent autotuning (TensorRT-style builders) must ship the
        # autotuning config or the build is not reproducible.
        if r.get("target_dependent_autotuning") and not r.get("autotuning_config"):
            errors.append("%s: target_dependent_autotuning is true but autotuning_config "
                          "is missing -- required for TensorRT-style builders" % ctx)


def _check_safety_evidence(pkg, profile, errors, warnings):
    rules = HARDENED_PROFILES[profile]
    ev = pkg.get("safety_evidence", {})
    if not isinstance(ev, dict):
        errors.append("safety_evidence: object required")
        ev = {}

    def _doc_present(key):
        d = ev.get(key)
        return isinstance(d, dict) and d.get("name") and d.get("version")

    if rules["requires_security_manual"] and not _doc_present("security_manual_reference"):
        errors.append("safety_evidence: profile '%s' requires a security_manual_reference "
                      "(name + version)" % profile)
    if rules["requires_safety_manual"] and not _doc_present("safety_manual_reference"):
        errors.append("safety_evidence: profile '%s' requires a safety_manual_reference "
                      "(name + version)" % profile)
    if rules["requires_tool_qualification"] and not _doc_present("tool_qualification_reference"):
        errors.append("safety_evidence: profile '%s' requires a tool_qualification_reference "
                      "(name + version)" % profile)

    # MLPerf does not certify safety; certification evidence is optional but, when
    # present, each entry must name a standard and a status.
    for i, c in enumerate(ev.get("certification_evidence", []) or []):
        ctx = "safety_evidence.certification_evidence[%d]" % i
        _require(c, "standard", ctx, errors)
        _require(c, "status", ctx, errors)

    if not ev.get("hardware_qualification"):
        warnings.append("safety_evidence: no hardware_qualification listed (e.g. AEC-Q100 "
                        "is expected for all profiles)")


# ---------------------------------------------------------------------------
# Version -> models. Resolved from the base checker's MODEL_CONFIG when available
# so we never duplicate the benchmark list; falls back to a static copy if the
# base module cannot be imported (keeps this file independently testable).
# ---------------------------------------------------------------------------

_FALLBACK_MODELS = {
    "v0.5": {"bevformer", "deeplabv3plus", "ssd"},
    "v1.0": {"bevformer", "deeplabv3plus", "ssd", "llama3_1-8b", "uniad"},
}


def _models_for_version(version):
    try:
        import submission_checker  # noqa: WPS433 (optional, integration-time)
        cfg = submission_checker.MODEL_CONFIG.get(version)
        if cfg and "models" in cfg:
            return set(cfg["models"])
    except Exception:  # pragma: no cover - import path only available in-tree
        pass
    return _FALLBACK_MODELS.get(version)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_hardened_package(pkg, system_desc_id=None):
    """Validate a parsed hardened package dict.

    Returns ``(is_valid, errors, warnings)``. ``is_valid`` is True iff there are no
    errors. The caller decides how to surface warnings.
    """
    errors = []
    warnings = []

    if not isinstance(pkg, dict):
        return False, ["hardened package: top-level value must be a JSON object"], []

    version = pkg.get("hardened_package_version")
    if version not in VALID_PACKAGE_VERSIONS:
        errors.append("hardened_package_version '%s' not in %s"
                      % (version, VALID_PACKAGE_VERSIONS))
        # Without a version we cannot resolve model lists; keep going but model checks
        # will use the fallback / be skipped.

    declared_id = pkg.get("system_desc_id")
    if not declared_id:
        errors.append("system_desc_id: field is missing")
    elif system_desc_id is not None and declared_id != system_desc_id:
        errors.append("system_desc_id '%s' does not match the system description "
                      "filename '%s'" % (declared_id, system_desc_id))

    profile = pkg.get("hardened_profile")
    if profile not in HARDENED_PROFILES:
        errors.append("hardened_profile '%s' not in %s"
                      % (profile, sorted(HARDENED_PROFILES)))

    _check_inventory(pkg, errors, warnings)
    _check_bom(pkg, errors, warnings)
    _check_oss(pkg, errors, warnings)
    _check_proprietary(pkg, errors, warnings)
    _check_licensing(pkg, errors, warnings)
    _check_reproducibility(pkg, version, errors, warnings)
    if profile in HARDENED_PROFILES:
        _check_safety_evidence(pkg, profile, errors, warnings)

    return (len(errors) == 0), errors, warnings


def check_hardened_package(name, hardened_json_path, system_desc_id=None):
    """File-level wrapper that mirrors ``check_system_desc_id`` in submission_checker.py:
    logs errors/warnings and returns a single boolean so it can be dropped into
    ``check_results_dir`` right after the ``status == 'hardened'`` branch.

    ``name`` is the human-readable results path used in the base checker's messages.
    """
    if not os.path.exists(hardened_json_path):
        log.error("%s: hardened submission requires %s", name, hardened_json_path)
        return False

    try:
        with open(hardened_json_path) as f:
            pkg = json.load(f)
    except (ValueError, OSError) as exc:
        log.error("%s: cannot parse hardened package %s (%s)",
                  name, hardened_json_path, exc)
        return False

    is_valid, errors, warnings = validate_hardened_package(pkg, system_desc_id)
    for w in warnings:
        log.warning("%s: %s", name, w)
    for e in errors:
        log.error("%s: %s", name, e)
    return is_valid


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validate an MLPerf Automotive Hardened Category package.")
    parser.add_argument("package", help="path to <system_desc>_hardened.json")
    parser.add_argument("--system-desc-id",
                        help="expected system_desc_id (basename of the systems json)")
    args = parser.parse_args()

    ok = check_hardened_package(args.package, args.package, args.system_desc_id)
    if ok:
        log.info("%s: hardened package is VALID", args.package)
    sys.exit(0 if ok else 1)
