"""Hardened Category checker for MLPerf Automotive submissions.

The Hardened Category requires submitters to disclose the full build and
runtime software stack used to produce and execute the deployable artifact,
along with licensing, an exact reproducibility recipe, and safety evidence.

This module validates the hardened metadata document that must accompany any
submission whose system status is "hardened" (see VALID_AVAILABILITIES in
submission_checker.py).

It can be run standalone against a single metadata JSON file or an entire
submission tree, and it is importable so that submission_checker.py can call
check_hardened_for_system() while walking the results directory.

Metadata file convention (per system):
    <division>/<submitter>/systems/<system_desc>_hardened.json

Only the discovery helpers below assume that path; the validators themselves are
path-agnostic, so if the WG prefers per-model granularity, only
find_hardened_metadata_for_system() and _iter_metadata_files() need to change.

Expected top-level schema (all sections required to be present):

    build_toolchain_bom          array of {component_name, version, role,
                                  configuration_flags}
    runtime_execution_bom        array of {component_name, component_type,
                                  version, configuration}
    oss_components               array of {component_name, version_or_commit,
                                  local_patches | patch_disclosure{...}}
    proprietary_components       array of {component_name, owner,
                                  version_or_build_id, licensing,
                                  distribution_model,
                                  internal_identifier_or_checksum,
                                  available_to_external_customers:bool}
    license_summary              array of {component_name, license_family,
                                  binary_redistribution_allowed:bool,
                                  source_redistribution_allowed:bool,
                                  additional_terms:str|null}
    mlperf_automotive_repo_reference  {implementation_repo_url,
                                  results_repo_url, commit_hash}
    model_compilation_recipe     {steps[], compiler_config,
                                  quantization_description, graph_optimizations,
                                  autotuning_config{...}}
    safety_manual_reference      {name, version} | null
    certification_evidence       str | object | array | null
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hardened")

HARDENED_STATUS = "hardened"
HARDENED_METADATA_SUFFIX = "_hardened.json"

SHA256_RE = re.compile(r"^[A-Fa-f0-9]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

# Values that signal patches exist but are deliberately not shared.
PATCHES_UNSHARED = {None, "", "not_shared", "unshared", "unavailable"}

# Top-level sections every hardened metadata document must contain.
REQUIRED_SECTIONS = [
    "build_toolchain_bom",
    "runtime_execution_bom",
    "oss_components",
    "proprietary_components",
    "license_summary",
    "mlperf_automotive_repo_reference",
    "model_compilation_recipe",
    "safety_manual_reference",
    "certification_evidence",
]

# Per-object required fields.
BUILD_TOOLCHAIN_STR_FIELDS = ["component_name", "version", "role"]
RUNTIME_EXECUTION_STR_FIELDS = ["component_name", "component_type", "version"]
PROPRIETARY_STR_FIELDS = [
    "component_name",
    "owner",
    "version_or_build_id",
    "licensing",
    "distribution_model",
    "internal_identifier_or_checksum",
]
LICENSE_STR_FIELDS = ["component_name", "license_family"]
LICENSE_BOOL_FIELDS = [
    "binary_redistribution_allowed",
    "source_redistribution_allowed",
]
REPO_REFERENCE_FIELDS = [
    "implementation_repo_url",
    "results_repo_url",
    "commit_hash",
]
OSS_PATCH_DISCLOSURE_FIELDS = [
    "component_boundary",
    "version_or_build_id",
    "inputs_outputs",
    "affects_kernel_or_tactic_selection",
]
AUTOTUNING_CONFIG_FIELDS = [
    "search_algorithm",
    "iteration_budget",
    "tactic_sources",
    "rng_seed",
]

# Recommended (not required) runtime component_type coverage. Substring match.
RECOMMENDED_RUNTIME_TYPES = ["firmware", "driver", "os", "runtime", "harness"]


class Report:
    """Accumulates errors and warnings for one metadata document."""

    def __init__(self, fname):
        self.fname = fname
        self.errors = []
        self.warnings = []

    def error(self, msg, *args):
        text = (msg % args) if args else msg
        self.errors.append(text)
        log.error("%s: %s", self.fname, text)

    def warning(self, msg, *args):
        text = (msg % args) if args else msg
        self.warnings.append(text)
        log.warning("%s: %s", self.fname, text)

    @property
    def is_valid(self):
        return len(self.errors) == 0


def _is_str(v):
    return isinstance(v, str) and v.strip() != ""


def _is_bool(v):
    return isinstance(v, bool)


def _is_list(v):
    return isinstance(v, list)


def _is_obj(v):
    return isinstance(v, dict)


def validate_array_section(report, metadata, key, per_item_validator):
    """Validate a top-level array-of-objects section."""
    if key not in metadata:
        return  # already reported by the top-level presence check
    value = metadata.get(key)
    if not _is_list(value):
        report.error("'%s' must be an array", key)
        return
    if len(value) == 0:
        report.error("'%s' must not be empty", key)
        return
    for idx, item in enumerate(value):
        ctx = "%s[%d]" % (key, idx)
        if not _is_obj(item):
            report.error("%s must be an object", ctx)
            continue
        per_item_validator(report, item, ctx)


def _validate_build_toolchain_item(report, item, ctx):
    for f in BUILD_TOOLCHAIN_STR_FIELDS:
        if not _is_str(item.get(f)):
            report.error("%s missing/invalid '%s'", ctx, f)
    if "configuration_flags" not in item:
        report.error("%s missing 'configuration_flags' (use {} or [] if none)", ctx)


def _validate_runtime_execution_item(report, item, ctx):
    for f in RUNTIME_EXECUTION_STR_FIELDS:
        if not _is_str(item.get(f)):
            report.error("%s missing/invalid '%s'", ctx, f)
    if "configuration" not in item:
        report.error("%s missing 'configuration' (use {} or [] if none)", ctx)


def _validate_oss_item(report, item, ctx):
    if not _is_str(item.get("component_name")):
        report.error("%s missing/invalid 'component_name'", ctx)
    if not _is_str(item.get("version_or_commit")):
        report.error("%s missing/invalid 'version_or_commit'", ctx)

    # Patches are either shared (local_patches present and not a "not shared"
    # marker) or, if they cannot be shared, a patch_disclosure block is
    # mandatory describing the component boundary and behavioural impact.
    local = item.get("local_patches")
    patches_shared = "local_patches" in item and local not in PATCHES_UNSHARED
    if not patches_shared:
        disclosure = item.get("patch_disclosure")
        if not _is_obj(disclosure):
            report.error(
                "%s patches not shared: 'patch_disclosure' object is required",
                ctx,
            )
            return
        for f in OSS_PATCH_DISCLOSURE_FIELDS:
            if f not in disclosure or disclosure[f] is None:
                report.error("%s.patch_disclosure missing '%s'", ctx, f)
        aff = disclosure.get("affects_kernel_or_tactic_selection")
        if "affects_kernel_or_tactic_selection" in disclosure and not _is_bool(aff):
            report.error(
                "%s.patch_disclosure 'affects_kernel_or_tactic_selection' must be boolean",
                ctx,
            )


def _validate_proprietary_item(report, item, ctx):
    for f in PROPRIETARY_STR_FIELDS:
        if not _is_str(item.get(f)):
            report.error("%s missing/invalid '%s'", ctx, f)
    if "available_to_external_customers" not in item:
        report.error("%s missing 'available_to_external_customers'", ctx)
    elif not _is_bool(item["available_to_external_customers"]):
        report.error(
            "%s 'available_to_external_customers' must be boolean", ctx
        )


def _validate_license_item(report, item, ctx):
    for f in LICENSE_STR_FIELDS:
        if not _is_str(item.get(f)):
            report.error("%s missing/invalid '%s'", ctx, f)
    for f in LICENSE_BOOL_FIELDS:
        if f not in item:
            report.error("%s missing '%s'", ctx, f)
        elif not _is_bool(item[f]):
            report.error("%s '%s' must be boolean", ctx, f)
    # additional_terms is required to be present but may be null (meaning none).
    if "additional_terms" not in item:
        report.error("%s missing 'additional_terms' (use null if none)", ctx)


def validate_repo_reference(report, metadata):
    key = "mlperf_automotive_repo_reference"
    if key not in metadata:
        return
    ref = metadata.get(key)
    if not _is_obj(ref):
        report.error("'%s' must be an object", key)
        return
    for f in REPO_REFERENCE_FIELDS:
        if not _is_str(ref.get(f)):
            report.error("%s missing/invalid '%s'", key, f)
    commit = ref.get("commit_hash")
    if _is_str(commit) and not COMMIT_RE.match(commit.strip()):
        report.warning(
            "%s.commit_hash '%s' does not look like a git commit hash",
            key,
            commit,
        )


def validate_autotuning(report, recipe):
    """Enforce the autotuning reproducibility hard rule.

    If autotuning is enabled, both a search configuration and a hash-pinned
    timing cache are mandatory. The configuration alone is insufficient: re-
    running a search re-benchmarks kernels and may select different tactics
    under measurement noise, so the build is not reproducible without the
    pinned cache.
    """
    at = recipe.get("autotuning_config")
    if not _is_obj(at):
        report.error(
            "model_compilation_recipe.autotuning_config must be an object with "
            "an 'enabled' flag (use {\"enabled\": false} if no autotuning)"
        )
        return
    enabled = at.get("enabled")
    if not _is_bool(enabled):
        report.error("autotuning_config.enabled must be present and boolean")
        return
    if not enabled:
        return  # No further requirements when autotuning is off.

    # HARD RULE: enabled => config + hash-pinned timing_cache both required.
    cfg = at.get("config")
    if not _is_obj(cfg):
        report.error(
            "autotuning enabled: autotuning_config.config object is required"
        )
    else:
        for f in AUTOTUNING_CONFIG_FIELDS:
            if f not in cfg or cfg[f] is None:
                report.error("autotuning_config.config missing '%s'", f)

    cache = at.get("timing_cache")
    if not _is_obj(cache):
        report.error(
            "autotuning enabled: a pinned autotuning_config.timing_cache is "
            "required (the build is not reproducible without it)"
        )
        return
    if not _is_str(cache.get("uri")):
        report.error("autotuning_config.timing_cache missing 'uri'")
    sha = cache.get("sha256")
    if not _is_str(sha) or not SHA256_RE.match(sha.strip()):
        report.error(
            "autotuning_config.timing_cache 'sha256' must be a 64-char hex "
            "digest (a bare URI to a mutable artifact is not reproducible)"
        )
    if "builder_versions" not in cache or cache["builder_versions"] is None:
        report.error(
            "autotuning_config.timing_cache missing 'builder_versions' "
            "(builder/library versions that key the cache)"
        )


def validate_model_compilation_recipe(report, metadata):
    key = "model_compilation_recipe"
    if key not in metadata:
        return
    recipe = metadata.get(key)
    if not _is_obj(recipe):
        report.error("'%s' must be an object", key)
        return

    steps = recipe.get("steps")
    if not _is_list(steps) or len(steps) == 0:
        report.error("%s.steps must be a non-empty ordered array", key)
    else:
        for i, s in enumerate(steps):
            if not _is_str(s):
                report.error(
                    "%s.steps[%d] must be a string or script URI", key, i
                )

    for f in ["compiler_config", "quantization_description", "graph_optimizations"]:
        if f not in recipe or recipe[f] is None:
            report.error("%s missing required field '%s'", key, f)

    validate_autotuning(report, recipe)


def validate_safety_evidence(report, metadata):
    # Both sections are required to be present (checked at top level). Their
    # values may be null/empty when not applicable, but we surface a warning so
    # the omission is a conscious one for a production-representative category.
    if "safety_manual_reference" in metadata:
        smr = metadata["safety_manual_reference"]
        if smr in (None, "", {}, []):
            report.warning(
                "safety_manual_reference is empty; provide name and version if "
                "a safety manual applies"
            )
        elif _is_obj(smr):
            for f in ["name", "version"]:
                if not _is_str(smr.get(f)):
                    report.warning("safety_manual_reference missing '%s'", f)

    if "certification_evidence" in metadata:
        ce = metadata["certification_evidence"]
        if ce in (None, "", {}, []):
            report.warning(
                "certification_evidence is empty; include any evidence of "
                "certification attempts or progress"
            )


def _check_runtime_type_coverage(report, metadata):
    items = metadata.get("runtime_execution_bom")
    if not _is_list(items):
        return
    seen = set()
    for item in items:
        if _is_obj(item):
            t = (item.get("component_type") or "").lower()
            for rec in RECOMMENDED_RUNTIME_TYPES:
                if rec in t:
                    seen.add(rec)
    missing = [t for t in RECOMMENDED_RUNTIME_TYPES if t not in seen]
    if missing:
        report.warning(
            "runtime_execution_bom does not obviously cover recommended "
            "component types: %s",
            ", ".join(missing),
        )


def check_hardened_metadata(metadata, fname="<metadata>"):
    """Validate an already-loaded hardened metadata dict. Returns a Report."""
    report = Report(fname)
    if not _is_obj(metadata):
        report.error("hardened metadata must be a JSON object")
        return report

    for section in REQUIRED_SECTIONS:
        if section not in metadata:
            report.error("missing required section '%s'", section)

    validate_array_section(
        report, metadata, "build_toolchain_bom", _validate_build_toolchain_item
    )
    validate_array_section(
        report, metadata, "runtime_execution_bom", _validate_runtime_execution_item
    )
    validate_array_section(
        report, metadata, "oss_components", _validate_oss_item
    )
    validate_array_section(
        report, metadata, "proprietary_components", _validate_proprietary_item
    )
    validate_array_section(
        report, metadata, "license_summary", _validate_license_item
    )
    validate_repo_reference(report, metadata)
    validate_model_compilation_recipe(report, metadata)
    validate_safety_evidence(report, metadata)
    _check_runtime_type_coverage(report, metadata)

    return report


def check_hardened_metadata_file(fname):
    """Load and validate a single hardened metadata JSON file."""
    try:
        with open(fname, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, ValueError) as e:
        report = Report(fname)
        report.error("could not load metadata JSON: %s", e)
        return report
    return check_hardened_metadata(metadata, fname)


def find_hardened_metadata_for_system(systems_dir, system_desc):
    candidate = os.path.join(
        systems_dir, system_desc + HARDENED_METADATA_SUFFIX
    )
    return candidate if os.path.exists(candidate) else None


def check_hardened_for_system(system_json, systems_dir, system_desc, name):
    """Integration hook for submission_checker.py.

    Returns True if the system is not hardened (nothing to check) or its
    hardened metadata is valid; False otherwise. Intended to be called from
    check_results_dir() right after the system_desc JSON is loaded, e.g.:

        systems_dir = os.path.join(division, submitter, "systems")
        if not check_hardened_for_system(
            system_json, systems_dir, system_desc, name
        ):
            results[name] = None
            continue
    """
    status = (system_json.get("status") or "").lower()
    if status != HARDENED_STATUS:
        return True
    metadata_file = find_hardened_metadata_for_system(systems_dir, system_desc)
    if metadata_file is None:
        log.error(
            "%s has status 'hardened' but no metadata file %s",
            name,
            os.path.join(systems_dir, system_desc + HARDENED_METADATA_SUFFIX),
        )
        return False
    report = check_hardened_metadata_file(metadata_file)
    return report.is_valid


def _iter_metadata_files(input_path):
    if os.path.isfile(input_path):
        yield input_path
        return
    for dirpath, _dirs, files in os.walk(input_path):
        if os.path.basename(dirpath) == "systems":
            for f in files:
                if f.endswith(HARDENED_METADATA_SUFFIX):
                    yield os.path.join(dirpath, f)


def main():
    parser = argparse.ArgumentParser(
        description="MLPerf Automotive Hardened Category checker"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="hardened metadata JSON file or a submission directory to walk",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as failures",
    )
    args = parser.parse_args()

    files = list(_iter_metadata_files(args.input))
    if not files:
        log.error("no hardened metadata files found under %s", args.input)
        return 1

    total_errors = 0
    total_warnings = 0
    for f in files:
        report = check_hardened_metadata_file(f)
        total_errors += len(report.errors)
        total_warnings += len(report.warnings)
        if report.is_valid:
            log.info("%s: OK (%d warnings)", f, len(report.warnings))
        else:
            log.error("%s: %d error(s)", f, len(report.errors))

    log.info("---")
    log.info(
        "Hardened metadata checked=%d, errors=%d, warnings=%d",
        len(files),
        total_errors,
        total_warnings,
    )
    if total_errors > 0 or (args.strict and total_warnings > 0):
        log.error("SUMMARY: hardened metadata has issues")
        return 1
    log.info("SUMMARY: hardened metadata looks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
