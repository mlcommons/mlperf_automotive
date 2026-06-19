"""Tests for hardened_checker.

Run: python3 -m unittest mlperf_hardened.test_hardened_checker  (from repo root)
 or: python3 test_hardened_checker.py                          (from this dir)

Covers: the three valid example packages, every cross-field rule the schema can't
express, and (optionally) JSON-Schema conformance if jsonschema happens to be installed.
"""

import copy
import json
import os
import unittest

import hardened_checker as hc

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "examples")
SCHEMA = os.path.join(HERE, "schema", "hardened_package.schema.json")


def load(name):
    with open(os.path.join(EX, name)) as f:
        return json.load(f)


class ValidExamples(unittest.TestCase):
    def test_adas_example_valid(self):
        ok, errors, _ = hc.validate_hardened_package(load("drive_orin_adas_hardened.json"),
                                                      "drive_orin_adas")
        self.assertTrue(ok, "unexpected errors: %s" % errors)

    def test_ivi_example_valid(self):
        ok, errors, _ = hc.validate_hardened_package(load("ivi_qm_hardened.json"),
                                                      "ivi_cockpit_qm")
        self.assertTrue(ok, "unexpected errors: %s" % errors)

    def test_invalid_example_fails(self):
        ok, errors, _ = hc.validate_hardened_package(load("invalid_adas_hardened.json"),
                                                      "broken_adas")
        self.assertFalse(ok)
        # spot-check that the headline rules each fired
        blob = " | ".join(errors)
        self.assertIn("safety_manual_reference", blob)
        self.assertIn("tool_qualification_reference", blob)
        self.assertIn("calibration_dataset_description", blob)
        self.assertIn("run_command", blob)
        self.assertIn("autotuning_config", blob)
        self.assertIn("commit_hash", blob)
        self.assertIn("patch", blob)


class CrossFieldRules(unittest.TestCase):
    def setUp(self):
        self.pkg = load("drive_orin_adas_hardened.json")

    def _expect_error_containing(self, pkg, needle):
        ok, errors, _ = hc.validate_hardened_package(pkg, pkg.get("system_desc_id"))
        self.assertFalse(ok)
        self.assertTrue(any(needle in e for e in errors),
                        "expected an error containing %r, got %s" % (needle, errors))

    def test_profile_must_be_known(self):
        self.pkg["hardened_profile"] = "L5"
        self._expect_error_containing(self.pkg, "hardened_profile")

    def test_system_desc_id_must_match(self):
        ok, errors, _ = hc.validate_hardened_package(self.pkg, "different_id")
        self.assertFalse(ok)
        self.assertTrue(any("does not match" in e for e in errors), errors)

    def test_adas_requires_safety_manual(self):
        del self.pkg["safety_evidence"]["safety_manual_reference"]
        self._expect_error_containing(self.pkg, "safety_manual_reference")

    def test_ivi_does_not_require_safety_manual(self):
        ivi = load("ivi_qm_hardened.json")
        self.assertNotIn("safety_manual_reference", ivi.get("safety_evidence", {}))
        ok, errors, _ = hc.validate_hardened_package(ivi, "ivi_cockpit_qm")
        self.assertTrue(ok, errors)

    def test_quantizer_requires_calibration(self):
        for c in self.pkg["software_bill_of_materials"]["build_toolchain_bom"]:
            if c["role"] == "quantizer":
                c.pop("calibration_dataset_description", None)
        self._expect_error_containing(self.pkg, "calibration_dataset_description")

    def test_application_requires_run_command(self):
        for c in self.pkg["software_bill_of_materials"]["runtime_execution_bom"]:
            if c["component_type"] == "application":
                c.pop("run_command", None)
        self._expect_error_containing(self.pkg, "run_command")

    def test_autotuning_requires_config(self):
        self.pkg["reproducibility"]["model_compilation_recipe"][0].pop("autotuning_config")
        self._expect_error_containing(self.pkg, "autotuning_config")

    def test_no_autotuning_no_config_is_fine(self):
        r = self.pkg["reproducibility"]["model_compilation_recipe"][0]
        r["target_dependent_autotuning"] = False
        r.pop("autotuning_config", None)
        ok, errors, _ = hc.validate_hardened_package(self.pkg, "drive_orin_adas")
        self.assertTrue(ok, errors)

    def test_audit_identifier_must_pin_something(self):
        self.pkg["production_intended_inventory"][0]["identifier"] = {}
        self._expect_error_containing(self.pkg, "identifier")

    def test_bad_checksum_format_rejected(self):
        self.pkg["production_intended_inventory"][0]["identifier"] = {
            "binary_checksum": "deadbeef"}
        self._expect_error_containing(self.pkg, "binary_checksum")

    def test_patch_without_disclosure_rejected(self):
        for c in self.pkg["oss_components"]:
            if c["component_name"] == "ONNX Runtime":
                c.pop("patch_feature_description", None)
                c.pop("patch_reference", None)
        self._expect_error_containing(self.pkg, "patch")

    def test_unknown_model_is_warning_not_error(self):
        self.pkg["reproducibility"]["model_compilation_recipe"][0]["model"] = "mystery_net"
        ok, errors, warnings = hc.validate_hardened_package(self.pkg, "drive_orin_adas")
        self.assertTrue(ok, errors)
        self.assertTrue(any("mystery_net" in w for w in warnings), warnings)

    def test_missing_required_top_section(self):
        del self.pkg["software_bill_of_materials"]
        self._expect_error_containing(self.pkg, "software_bill_of_materials")

    def test_benchmark_only_inventory_warns(self):
        for c in self.pkg["production_intended_inventory"]:
            c["intended_for_production"] = False
        ok, errors, warnings = hc.validate_hardened_package(self.pkg, "drive_orin_adas")
        self.assertTrue(ok, errors)
        self.assertTrue(any("intended_for_production" in w for w in warnings), warnings)


class SchemaConformance(unittest.TestCase):
    """Only runs if jsonschema is installed; the production checker does not need it."""

    def setUp(self):
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest("jsonschema not installed (optional)")
        with open(SCHEMA) as f:
            self.schema = json.load(f)

    def test_valid_examples_match_schema(self):
        import jsonschema
        for name in ("drive_orin_adas_hardened.json", "ivi_qm_hardened.json"):
            jsonschema.validate(load(name), self.schema)

    def test_invalid_example_violates_schema_or_checker(self):
        # The invalid example is crafted to fail the *checker*; it may still be
        # structurally schema-valid (the cross-field rules are checker-only). This
        # test just asserts the schema itself is a valid draft-07 document.
        import jsonschema
        jsonschema.Draft7Validator.check_schema(self.schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)
