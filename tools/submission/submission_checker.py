"""A checker for MLPerf Inference submissions from v4.1 onwards (for checking older submissions please use the submission checker from the respective release)
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import datetime
import json
import logging
import os
import re
import sys

from glob import glob

from log_parser import MLPerfLog

# pylint: disable=missing-docstring

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

submission_checker_dir = os.path.dirname(os.path.realpath(__file__))

MODEL_CONFIG = {
    "v0.5": {
        "models": [
            "bevformer",
            "deeplabv3plus",
            "ssd",
        ],
        "required-scenarios-adas": {
            "bevformer": ["SingleStream"],
            "deeplabv3plus": ["SingleStream"],
            "ssd": ["SingleStream"]
        },
        "optional-scenarios-adas": {
            "bevformer": ["ConstantStream"],
            "deeplabv3plus": ["ConstantStream"],
            "ssd": ["ConstantStream"]
        },
        "accuracy-target": {
            "bevformer": ("mAP_3D", .2683556 * 0.99),
            "deeplabv3plus": ("mIOU", .924355 * 0.999),
            "ssd": ("mAP", .7179 * 0.999)
        },
        "accuracy-upper-limit": {

        },
        "accuracy-delta-perc": {

        },
        "performance-sample-count": {
            "bevformer": 256,
            "deeplabv3plus": 128,
            "ssd": 128
        },
        # model_mapping.json is expected in the root directory of the
        # submission folder for open submissions and so the below dictionary is
        # not really needed
        "model_mapping": {
            # map model names to the official mlperf model class
            "SSD": "ssd",
            "BEVFORMER": "bevformer",
            "DEEPLABV3PLUS": "deeplabv3plus",
        },
        "seeds": {
            "qsl_rng_seed": 1575625098,
            "sample_index_rng_seed": 2227286192,
            "schedule_rng_seed": 3495234579,
        },
        "ignore_errors": [],
        "latency-constraint": {},
        "min-queries": {
            "bevformer": {
                "ConstantStream": 100000,
                "SingleStream": 6636,
            },
            "deeplabv3plus": {
                "ConstantStream": 100000,
                "SingleStream": 6636,
            },
            "ssd": {
                "ConstantStream": 100000,
                "SingleStream": 6636,
            }
        },
    },
}

VALID_DIVISIONS = ["open", "closed", "network"]
VALID_AVAILABILITIES = [
    "hardened",
    "development",
    "engineering_samples",
    "presilicon"]
REQUIRED_PERF_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
OPTIONAL_PERF_FILES = ["mlperf_log_accuracy.json"]
REQUIRED_PERF_POWER_FILES = ["spl.txt"]
REQUIRED_POWER_FILES = [
    "client.json",
    "client.log",
    "ptd_logs.txt",
    "server.json",
    "server.log",
]
REQUIRED_ACC_FILES = [
    "mlperf_log_summary.txt",
    "mlperf_log_detail.txt",
    "accuracy.txt",
    "mlperf_log_accuracy.json",
]
REQUIRED_MEASURE_FILES = ["user.conf", "README.md"]
REQUIRED_POWER_MEASURE_FILES = ["analyzer_table.*", "power_settings.*"]
MS_TO_NS = 1000 * 1000
S_TO_MS = 1000
FILE_SIZE_LIMIT_MB = 50
MB_TO_BYTES = 1024 * 1024
MAX_ACCURACY_LOG_SIZE = 10 * 1024
OFFLINE_MIN_SPQ = 24576
TEST_DURATION_MS_PRE_1_0 = 60000
TEST_DURATION_MS = 600000
REQUIRED_COMP_PER_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
REQUIRED_TEST01_ACC_FILES_1 = ["mlperf_log_accuracy.json", "accuracy.txt"]
REQUIRED_TEST01_ACC_FILES = REQUIRED_TEST01_ACC_FILES_1 + [
    "baseline_accuracy.txt",
    "compliance_accuracy.txt",
]

OFFLINE_MIN_SPQ_SINCE_V4 = {
    # TODO: Update or remove
    "bevformer": 1024,
    "deeplabv3plus": 1024,
    "ssd": 1024
}

SCENARIO_MAPPING = {
    "singlestream": "SingleStream",
    "multistream": "MultiStream",
    "server": "ConstantStream",
    "constantstream": "ConstantStream",
    "offline": "Offline",
}

RESULT_FIELD = {
    "Offline": "Samples per second",
    "SingleStream": "90th percentile latency (ns)",
    "MultiStream": "Samples per query",
    "ConstantStream": "Scheduled samples per second",
}

RESULT_FIELD_NEW = {
    "v0.5": {
        "Offline": "result_samples_per_second",
        "SingleStream": "early_stopping_latency_ss",
        "MultiStream": "early_stopping_latency_ms",
        "ConstantStream": "result_completed_samples_per_sec",
    },
}

ACC_PATTERN = {
    "mAP": r"mAP:\s*([0-9.]+)",
    "mIOU": r"Mean IoU:\s*([0-9.]+)",
    "mAP_3D": r"mAP_3D:\s*([0-9.]+)",
}

SYSTEM_DESC_REQUIRED_FIELDS = [
    "division",
    "submitter",
    "status",
    "system_name",
    "number_of_nodes",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_storage_capacity",
    "host_storage_type",
    "accelerators_per_node",
    "accelerator_model_name",
    "accelerator_memory_capacity",
    "framework",
    "operating_system",
    "system_type",
    "other_software_stack",
    "host_processor_frequency",
    "host_processor_caches",
    "host_memory_configuration",
    "host_processor_interconnect",
    "host_networking",
    "host_networking_topology",
    "accelerator_frequency",
    "accelerator_host_interconnect",
    "accelerator_interconnect",
    "accelerator_interconnect_topology",
    "accelerator_memory_configuration",
    "accelerator_on-chip_memories",
    "cooling",
    "hw_notes",
    "sw_notes",
    "host_network_card_count",
    "system_type_detail",
]

SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS = [
    "division",
    "submitter",
    "system_type",
    "status",
    "system_name",
    "number_of_nodes",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_memory_configuration",
    "host_storage_capacity",
    "host_storage_type",
    "host_networking",
    "host_network_card_count",
    "host_networking_topology",
    "accelerators_per_node",
    "accelerator_model_name",
    "accelerator_memory_capacity",
    "accelerator_host_interconnect",
    "accelerator_memory_configuration",
    "accelerator_interconnect",
    "cooling",
    "framework",
    "operating_system",
    "other_software_stack",
]

SYSTEM_DESC_REQUIRED_FIELDS_POWER = [
    "power_management",
    "filesystem",
    "boot_firmware_version",
    "management_firmware_version",
    "other_hardware",
    "number_of_type_nics_installed",
    "nics_enabled_firmware",
    "nics_enabled_os",
    "nics_enabled_connected",
    "network_speed_mbit",
    "power_supply_quantity_and_rating_watts",
    "power_supply_details",
    "disk_drives",
    "disk_controllers",
    "system_power_only",
]

SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS_POWER = []

SYSTEM_DESC_IS_NETWORK_MODE = "is_network"
SYSTEM_DESC_REQUIRED_FIELDS_NETWORK_MODE = [
    SYSTEM_DESC_IS_NETWORK_MODE,
    "network_type",
    "network_media",
    "network_rate",
    "nic_loadgen",
    "number_nic_loadgen",
    "net_software_stack_loadgen",
    "network_protocol",
    "number_connections",
    "nic_sut",
    "number_nic_sut",
    "net_software_stack_sut",
    "network_topology",
]
NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME = "Network SUT"

SYSTEM_IMP_REQUIRED_FILES = [
    "input_data_types",
    "retraining",
    "starting_weights_filename",
    "weight_data_types",
    "weight_transformations",
]


class Config:
    """Select config value by mlperf version and submission type."""

    def __init__(
        self,
        version,
        extra_model_benchmark_map,
        ignore_uncommited=False,
        skip_power_check=False,
    ):
        self.base = MODEL_CONFIG.get(version)
        self.extra_model_benchmark_map = extra_model_benchmark_map
        self.version = version
        self.models = self.base["models"]
        self.seeds = self.base["seeds"]
        if self.base.get("test05_seeds"):
            self.test05_seeds = self.base["test05_seeds"]
        self.accuracy_target = self.base["accuracy-target"]
        self.accuracy_delta_perc = self.base["accuracy-delta-perc"]
        self.accuracy_upper_limit = self.base.get("accuracy-upper-limit", {})
        self.performance_sample_count = self.base["performance-sample-count"]
        self.latency_constraint = self.base.get("latency-constraint", {})
        self.min_queries = self.base.get("min-queries", {})
        self.required = None
        self.optional = None
        self.ignore_uncommited = ignore_uncommited
        self.skip_power_check = skip_power_check

    def set_type(self, submission_type):
        if submission_type == "adas":
            self.required = self.base["required-scenarios-adas"]
            self.optional = self.base["optional-scenarios-adas"]
        else:
            raise ValueError("invalid system type")

    def get_mlperf_model(self, model, extra_model_mapping=None):
        # preferred - user is already using the official name
        if model in self.models:
            return model

        # simple mapping, ie resnet50->resnet
        mlperf_model = self.base["model_mapping"].get(model)
        if mlperf_model:
            return mlperf_model

        # Custom mapping provided by the submitter
        if extra_model_mapping is not None:
            mlperf_model = extra_model_mapping.get(model)
            if mlperf_model:
                return mlperf_model

        # try to guess, keep this for backwards compatibility
        # TODO: Generalize this guess or remove it completely?

        if "mobilenet" in model or "efficientnet" in model or "resnet50" in model:
            model = "resnet"
        elif "bert-99.9" in model:
            model = "bert-99.9"
        elif "bert-99" in model:
            model = "bert-99"
        # map again
        mlperf_model = self.base["model_mapping"].get(model, model)
        return mlperf_model

    def get_required(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.required:
            return None
        return set(self.required[model])

    def get_optional(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.optional:
            return set()
        return set(self.optional[model])

    def get_accuracy_target(self, model):
        if model not in self.accuracy_target:
            raise ValueError("model not known: " + model)
        return self.accuracy_target[model]

    def get_accuracy_upper_limit(self, model):
        return self.accuracy_upper_limit.get(model, None)

    def get_performance_sample_count(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.performance_sample_count:
            raise ValueError("model not known: " + model)
        return self.performance_sample_count[model]

    def ignore_errors(self, line):
        for error in self.base["ignore_errors"]:
            if error in line:
                return True
        if (
            self.ignore_uncommited
            and ("ERROR : Loadgen built with uncommitted " "changes!") in line
        ):
            return True
        return False

    def get_min_query_count(self, model, scenario):
        model = self.get_mlperf_model(model)
        if model not in self.min_queries:
            raise ValueError("model not known: " + model)
        return self.min_queries[model].get(scenario)

    def get_delta_perc(self, model, metric):
        if model in self.accuracy_delta_perc:
            if metric in self.accuracy_delta_perc[model]:
                return self.accuracy_delta_perc[model][metric]

        more_accurate = model.find("99.9")
        if more_accurate == -1:
            required_delta_perc = 1
        else:
            required_delta_perc = 0.1
        return required_delta_perc

    def has_new_logging_format(self):
        return True

    def uses_early_stopping(self, scenario):
        return scenario in ["ConstantStream", "SingleStream", "MultiStream"]

    def requires_equal_issue(self, model, division):
        return (
            division in ["closed", "network"]
            and model
            in []
            and self.version not in ["v4.0", "v4.1"]
        )


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument(
        "--version",
        default="v0.5",
        choices=list(MODEL_CONFIG.keys()),
        help="mlperf version",
    )
    parser.add_argument("--submitter", help="filter to submitter")
    parser.add_argument(
        "--csv",
        default="summary.csv",
        help="csv file with results")
    parser.add_argument(
        "--extra-model-benchmark-map",
        help="File containing extra custom model mapping. It is assumed to be inside the folder open/<submitter>",
        default="model_mapping.json",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="extra debug output")
    parser.add_argument(
        "--submission-exceptions",
        action="store_true",
        help="ignore certain errors for submission",
    )
    parser.add_argument(
        "--skip-power-check",
        action="store_true",
        help="skips Power WG's check.py script on each power submission.",
    )
    parser.add_argument(
        "--skip-meaningful-fields-emptiness-check",
        action="store_true",
        help="skips the check of empty values in required measurement field values",
    )
    parser.add_argument(
        "--skip-check-power-measure-files",
        action="store_true",
        help="skips the check of required measure files for power runs",
    )
    parser.add_argument(
        "--skip-empty-files-check",
        action="store_true",
        help="skips the check of empty required files",
    )
    parser.add_argument(
        "--skip-extra-files-in-root-check",
        action="store_true",
        help="skips the check of extra files inside the root submission dir",
    )
    parser.add_argument(
        "--skip-extra-accuracy-files-check",
        action="store_true",
        help="skips the check of extra accuracy files like the images folder of SDXL",
    )
    parser.add_argument(
        "--scenarios-to-skip",
        help="Delimited list input of scenarios to skip. i.e. if you only have Offline results, pass in 'ConstantStream'",
        type=str,
    )
    args = parser.parse_args()
    return args


def list_dir(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(
        path) if os.path.isdir(os.path.join(path, f))]


def list_files(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]


def list_empty_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(
        path) if not dirs and not files]


def list_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(path)]


def list_files_recursively(*path):
    path = os.path.join(*path)
    return [
        os.path.join(dirpath, file)
        for dirpath, dirs, files in os.walk(path)
        for file in files
    ]


def check_extra_files(path, target_files):
    missing_files = []
    check_pass = True
    folders = list_dir(path)
    for dir in target_files.keys():
        if dir not in folders:
            check_pass = False
            missing_files.append(os.path.join(path, dir))
        else:
            files = [f.split(".")[0]
                     for f in list_files(os.path.join(path, dir))]
            for target_file in target_files[dir]:
                if target_file not in files:
                    check_pass = False
                    missing_files.append(
                        f"{os.path.join(path, dir, target_file)}.png")
            if "captions" not in files:
                missing_files.append(
                    f"{os.path.join(path, dir, 'captions.txt')}")
    return check_pass, missing_files


def split_path(m):
    return m.replace("\\", "/").split("/")


def get_boolean(s):
    if s is None:
        return False
    elif isinstance(s, bool):
        return s
    elif isinstance(s, str):
        return s.lower() == "true"
    elif isinstance(s, int):
        return bool(s)
    else:
        raise TypeError(
            f"Variable should be bool, string or int, got {type(s)} instead"
        )


def find_error_in_detail_log(config, fname):
    is_valid = True
    if not os.path.exists(fname):
        log.error("%s is missing", fname)
        is_valid = False
    else:
        mlperf_log = MLPerfLog(fname)
        if mlperf_log.has_error():
            if config.ignore_uncommited:
                has_other_errors = False
                for error in mlperf_log.get_errors():
                    if "Loadgen built with uncommitted changes!" not in error["value"]:
                        has_other_errors = True

            log.error("%s contains errors:", fname)
            for error in mlperf_log.get_errors():
                log.error("%s", error["value"])

            if not config.ignore_uncommited or has_other_errors:
                is_valid = False
    return is_valid


def get_accuracy_values(config, model):

    patterns = []
    acc_targets = []
    acc_types = []
    acc_limits = []
    up_patterns = []
    acc_limit_check = False

    target = config.get_accuracy_target(model)
    acc_upper_limit = config.get_accuracy_upper_limit(model)
    if acc_upper_limit is not None:
        for i in range(0, len(acc_upper_limit), 2):
            acc_type, acc_target = acc_upper_limit[i: i + 2]
            acc_limits.append(acc_target)
            up_patterns.append(ACC_PATTERN[acc_type])

    for i in range(0, len(target), 2):
        acc_type, acc_target = target[i: i + 2]
        patterns.append(ACC_PATTERN[acc_type])
        acc_targets.append(acc_target)
        acc_types.append(acc_type)

    return patterns, acc_targets, acc_types, acc_limits, up_patterns, acc_upper_limit


def check_accuracy_dir(config, model, path, verbose):
    is_valid = False
    all_accuracy_valid = True
    acc = None
    result_acc = {}
    hash_val = None
    target = config.get_accuracy_target(model)
    # acc_upper_limit = config.get_accuracy_upper_limit(model)
    patterns, acc_targets, acc_types, acc_limits, up_patterns, acc_upper_limit = get_accuracy_values(
        config, model)
    acc_limit_check = True

    acc_seen = [False for _ in acc_targets]

    with open(os.path.join(path, "accuracy.txt"), "r", encoding="utf-8") as f:
        for line in f:
            for i, (pattern, acc_target, acc_type) in enumerate(
                zip(patterns, acc_targets, acc_types)
            ):
                m = re.match(pattern, line)
                if m:
                    acc = m.group(1)
                m = re.match(r"^hash=([\w\d]+)$", line)
                if m:
                    hash_val = m.group(1)
                if acc is not None and float(acc) >= acc_target:
                    all_accuracy_valid &= True
                    acc_seen[i] = True
                elif acc is not None:
                    all_accuracy_valid = False
                    log.warning(
                        "%s accuracy not met: expected=%f, found=%s",
                        path,
                        acc_target,
                        acc,
                    )
                if acc:
                    result_acc[acc_type] = acc
                acc = None

            if acc_upper_limit is not None:
                for i, (pattern, acc_limit) in enumerate(
                        zip(up_patterns, acc_limits)):
                    m = re.match(pattern, line)
                    if m:
                        acc = m.group(1)
                    m = re.match(r"^hash=([\w\d]+)$", line)
                    if m:
                        hash_val = m.group(1)
                    if (
                        acc is not None
                        and acc_upper_limit is not None
                        and float(acc) > acc_limit
                    ):
                        acc_limit_check = False
                        log.warning(
                            "%s accuracy not met: upper limit=%f, found=%s",
                            path,
                            acc_limit,
                            acc,
                        )
                    acc = None
            if all(acc_seen) and hash_val:
                break
        is_valid = all_accuracy_valid & all(acc_seen)
        if acc_upper_limit is not None:
            is_valid &= acc_limit_check

    if not hash_val:
        log.error("%s not hash value for mlperf_log_accuracy.json", path)
        is_valid = False

    # check mlperf_log_accuracy.json
    fname = os.path.join(path, "mlperf_log_accuracy.json")
    if not os.path.exists(fname):
        log.error("%s is missing", fname)
        is_valid = False
    else:
        if os.stat(fname).st_size > MAX_ACCURACY_LOG_SIZE:
            log.error("%s is not truncated", fname)
            is_valid = False

    # check if there are any errors in the detailed log
    fname = os.path.join(path, "mlperf_log_detail.txt")
    if not find_error_in_detail_log(config, fname):
        is_valid = False

    return is_valid, result_acc


def get_performance_metric(
        config, model, path, scenario_fixed):
    # Assumes new logging format
    version = config.version

    fname = os.path.join(path, "mlperf_log_detail.txt")
    mlperf_log = MLPerfLog(fname)
    if (
        "result_validity" in mlperf_log.get_keys()
        and mlperf_log["result_validity"] == "VALID"
    ):
        is_valid = True
    scenario = mlperf_log["effective_scenario"]

    log.info("%s, %s", version, scenario)

    res = float(mlperf_log[RESULT_FIELD_NEW[version][scenario]])
    inferred = False
    if scenario_fixed != scenario:
        inferred, res = get_inferred_result(
            scenario_fixed, scenario, res, mlperf_log, config, False
        )

    return res


def check_performance_dir(
        config, model, path, scenario_fixed, division, system_json):
    is_valid = False
    rt = {}

    version = config.version
    # look for: Result is: VALID
    fname = os.path.join(path, "mlperf_log_detail.txt")
    mlperf_log = MLPerfLog(fname)
    if (
        "result_validity" in mlperf_log.get_keys()
        and mlperf_log["result_validity"] == "VALID"
    ):
        is_valid = True
    performance_sample_count = mlperf_log["effective_performance_sample_count"]
    qsl_rng_seed = mlperf_log["effective_qsl_rng_seed"]
    sample_index_rng_seed = mlperf_log["effective_sample_index_rng_seed"]
    schedule_rng_seed = mlperf_log["effective_schedule_rng_seed"]
    scenario = mlperf_log["effective_scenario"]
    constant_gen = mlperf_log["effective_server_constant_gen"]
    grouped_qsl = mlperf_log["effective_use_grouped_qsl"]
    target_latency_percentile = mlperf_log["effective_target_latency_percentile"]

    res = float(mlperf_log[RESULT_FIELD_NEW[version][scenario]])

    latency_99_percentile = mlperf_log["result_99.00_percentile_latency_ns"]
    latency_mean = mlperf_log["result_mean_latency_ns"]
    if scenario in ["MultiStream"]:
        latency_99_percentile = mlperf_log[
            "result_99.00_percentile_per_query_latency_ns"
        ]
        latency_mean = mlperf_log["result_mean_query_latency_ns"]
    min_query_count = mlperf_log["effective_min_query_count"]
    samples_per_query = mlperf_log["effective_samples_per_query"]
    min_duration = mlperf_log["effective_min_duration_ms"]
    equal_issue_used_check = (
        mlperf_log["effective_sample_concatenate_permutation"] == True
    )
    if not config.requires_equal_issue(model, division):
        equal_issue_used_check = True
    if not equal_issue_used_check:
        log.error(
            "%s requires equal issue mode (sample_concatenate_permutation), expected=true, found=false", path
        )
        is_valid = False

    sut_name = mlperf_log["sut_name"]

    # check if there are any errors in the detailed log
    fname = os.path.join(path, "mlperf_log_detail.txt")
    if not find_error_in_detail_log(config, fname):
        is_valid = False

    required_performance_sample_count = config.get_performance_sample_count(
        model)
    if performance_sample_count < required_performance_sample_count:
        log.error(
            "%s performance_sample_count, found %d, needs to be >= %d",
            fname,
            performance_sample_count,
            required_performance_sample_count,
        )
        is_valid = False

    config_seeds = config.seeds if "TEST05" not in fname else config.test05_seeds
    if qsl_rng_seed != config_seeds["qsl_rng_seed"]:
        log.error(
            "%s qsl_rng_seed is wrong, expected=%s, found=%s",
            fname,
            config_seeds["qsl_rng_seed"],
            qsl_rng_seed,
        )
        is_valid = False
    if sample_index_rng_seed != config_seeds["sample_index_rng_seed"]:
        log.error(
            "%s sample_index_rng_seed is wrong, expected=%s, found=%s",
            fname,
            config_seeds["sample_index_rng_seed"],
            sample_index_rng_seed,
        )
        is_valid = False
    if schedule_rng_seed != config_seeds["schedule_rng_seed"]:
        log.error(
            "%s schedule_rng_seed is wrong, expected=%s, found=%s",
            fname,
            config_seeds["schedule_rng_seed"],
            schedule_rng_seed,
        )
        is_valid = False
    if scenario == "ConstantStream" and not constant_gen:
        log.error(
            "%s constant_gen is set to false, expected=%s, found=%s",
            fname,
            True,
            constant_gen,
        )
        is_valid = False

    if not grouped_qsl and model in ["bevformer"]:
        log.error(
            "%s grouped_qsl is required but not used, expected=%s, found=%s",
            fname,
            True,
            constant_gen,
        )
        is_valid = False

    if target_latency_percentile != 0.999:
        log.error(
            "%s target_latency_percentile is required to be 0.999, expected=%s, found=%s",
            fname,
            "0.999",
            target_latency_percentile,
        )
        is_valid = False

    if scenario == "SingleStream" or scenario == "MultiStream":
        res /= MS_TO_NS

    # Check if the current scenario uses early stopping
    uses_early_stopping = config.uses_early_stopping(scenario)

    if uses_early_stopping:
        # check if early_stopping condition was met
        if not mlperf_log["early_stopping_met"]:
            early_stopping_result = mlperf_log["early_stopping_result"]
            log.error(
                "Early stopping condition was not met, msg=%s",
                early_stopping_result,
            )

        # If the scenario has a target latency (ConstantStream scenario), check
        # that the target latency that was passed to the early stopping
        # is less than the target latency.
        target_latency = config.latency_constraint.get(
            model, dict()).get(scenario)
        if target_latency:
            early_stopping_latency_ns = mlperf_log["effective_target_latency_ns"]
            log.info(
                "Target latency: %s, Early Stopping Latency: %s, Scenario: %s",
                target_latency,
                early_stopping_latency_ns,
                scenario,
            )
            if early_stopping_latency_ns > target_latency:
                log.error(
                    "%s Latency constraint with early stopping not met, expected=%s, found=%s",
                    fname,
                    target_latency,
                    early_stopping_latency_ns,
                )
                is_valid = False

    else:
        # check if the benchmark meets latency constraint
        target_latency = config.latency_constraint.get(
            model, dict()).get(scenario)
        log.info(
            "Target latency: %s, Latency: %s, Scenario: %s",
            target_latency,
            latency_99_percentile,
            scenario,
        )
        if target_latency:
            if latency_99_percentile > target_latency:
                log.error(
                    "%s Latency constraint not met, expected=%s, found=%s",
                    fname,
                    target_latency,
                    latency_99_percentile,
                )

    # Check Minimum queries were issued to meet test duration
    # Check if this run uses early stopping. If it does, get the
    # min_queries from the detail log, otherwise get this value
    # from the config
    if not uses_early_stopping:
        required_min_query_count = config.get_min_query_count(model, scenario)
        if required_min_query_count and min_query_count < required_min_query_count:
            log.error(
                "%s Required minimum Query Count not met by user config, Expected=%s, Found=%s",
                fname,
                required_min_query_count,
                min_query_count,
            )
            is_valid = False

    if scenario == "Offline" and (
            samples_per_query < OFFLINE_MIN_SPQ_SINCE_V4[model]):
        log.error(
            "%s Required minimum samples per query not met by user config, Expected=%s, Found=%s",
            fname,
            OFFLINE_MIN_SPQ_SINCE_V4[model],
            samples_per_query,
        )
        is_valid = False

    # Test duration of 600s is met
    required_min_duration = TEST_DURATION_MS

    if min_duration < required_min_duration:
        log.error(
            "%s Test duration less than 600s in user config. expected=%s, found=%s",
            fname,
            required_min_duration,
            min_duration,
        )
        is_valid = False

    inferred = False
    if scenario_fixed != scenario:
        inferred, res = get_inferred_result(
            scenario_fixed, scenario, res, mlperf_log, config, True
        )

    is_network_system, is_network_mode_valid = is_system_over_network(
        division, system_json, path
    )
    is_valid &= is_network_mode_valid
    if is_network_system:
        # for network mode verify the SUT name is valid, according to the rules
        # (must include "Network SUT" in name)
        if NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME not in sut_name:
            log.error(
                f"{fname} invalid sut name for network mode. expecting the substring '{NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME}' got '{sut_name}'"
            )
            is_valid = False

    return is_valid, res, inferred


def get_inferred_result(
    scenario_fixed, scenario, res, mlperf_log, config, log_error=False
):

    inferred = False
    # Check if current scenario (and version) uses early stopping
    uses_early_stopping = config.uses_early_stopping(scenario)

    latency_mean = mlperf_log["result_mean_latency_ns"]
    if scenario in ["MultiStream"]:
        latency_99_percentile = mlperf_log[
            "result_99.00_percentile_per_query_latency_ns"
        ]
        latency_mean = mlperf_log["result_mean_query_latency_ns"]
    samples_per_query = mlperf_log["effective_samples_per_query"]
    if scenario == "SingleStream":
        # qps_wo_loadgen_overhead is only used for inferring Offline from
        # SingleStream; only for old submissions
        qps_wo_loadgen_overhead = mlperf_log["result_qps_without_loadgen_overhead"]

    # special case for results inferred from different scenario
    if scenario_fixed in ["Offline"] and scenario in ["SingleStream"]:
        inferred = True
        res = qps_wo_loadgen_overhead

    if (scenario_fixed in ["Offline"]) and scenario in ["MultiStream"]:
        inferred = True
        res = samples_per_query * S_TO_MS / (latency_mean / MS_TO_NS)

    if (scenario_fixed in ["MultiStream"]) and scenario in ["SingleStream"]:
        inferred = True
        # samples_per_query does not match with the one reported in the logs
        # when inferring MultiStream from SingleStream
        samples_per_query = 8
        if uses_early_stopping:
            early_stopping_latency_ms = mlperf_log["early_stopping_latency_ms"]
            if early_stopping_latency_ms == 0 and log_error:
                log.error(
                    "Not enough samples were processed for early stopping to make an estimate"
                )
                is_valid = False
            res = (early_stopping_latency_ms * samples_per_query) / MS_TO_NS
        else:
            res = (latency_99_percentile * samples_per_query) / MS_TO_NS
    return inferred, res


def get_power_metric(config, scenario_fixed, log_path, is_valid, res):
    # parse the power logs
    server_timezone = datetime.timedelta(0)
    client_timezone = datetime.timedelta(0)

    detail_log_fname = os.path.join(log_path, "mlperf_log_detail.txt")
    mlperf_log = MLPerfLog(detail_log_fname)
    datetime_format = "%m-%d-%Y %H:%M:%S.%f"
    power_begin = (
        datetime.datetime.strptime(mlperf_log["power_begin"], datetime_format)
        + client_timezone
    )
    power_end = (
        datetime.datetime.strptime(mlperf_log["power_end"], datetime_format)
        + client_timezone
    )
    # Obtain the scenario also from logs to check if power is inferred
    scenario = mlperf_log["effective_scenario"]

    spl_fname = os.path.join(log_path, "spl.txt")
    power_list = []
    with open(spl_fname) as f:
        for line in f:
            if not line.startswith("Time"):
                continue
            timestamp = (
                datetime.datetime.strptime(line.split(",")[1], datetime_format)
                + server_timezone
            )
            if timestamp > power_begin and timestamp < power_end:
                value = float(line.split(",")[3])
                if value > 0:
                    power_list.append(float(line.split(",")[3]))

    if len(power_list) == 0:
        log.error(
            "%s has no power samples falling in power range: %s - %s",
            spl_fname,
            power_begin,
            power_end,
        )
        is_valid = False
    else:
        avg_power = sum(power_list) / len(power_list)
        power_duration = (power_end - power_begin).total_seconds()
        if scenario_fixed in ["Offline", "ConstantStream"]:
            # In Offline and ConstantStream scenarios, the power metric is in
            # W.
            power_metric = avg_power
            avg_power_efficiency = res / avg_power

        else:
            # In SingleStream and MultiStream scenarios, the power metric is in
            # mJ/query.
            assert scenario_fixed in [
                "MultiStream",
                "SingleStream",
            ], "Unknown scenario: {:}".format(scenario_fixed)

            num_queries = int(mlperf_log["result_query_count"])

            power_metric = avg_power * power_duration * 1000 / num_queries

            if scenario_fixed in ["SingleStream"]:
                samples_per_query = 1
            elif scenario_fixed in ["MultiStream"]:
                samples_per_query = 8

            if (scenario_fixed in ["MultiStream"]
                    ) and scenario in ["SingleStream"]:
                power_metric = (
                    avg_power * power_duration * samples_per_query * 1000 / num_queries
                )

            avg_power_efficiency = (samples_per_query * 1000) / power_metric

    return is_valid, power_metric, scenario, avg_power_efficiency


def check_power_dir(
    power_path,
    ranging_path,
    testing_path,
    scenario_fixed,
    power_res_ranging,
    power_res_testing,
    config,
):
    skip_power_check = config.skip_power_check

    is_valid = True
    power_metric = 0

    # check if all the required files are present
    required_files = REQUIRED_PERF_FILES + REQUIRED_PERF_POWER_FILES
    diff = files_diff(
        list_files(testing_path),
        required_files,
        OPTIONAL_PERF_FILES)
    if diff:
        log.error("%s has file list mismatch (%s)", testing_path, diff)
        is_valid = False
    diff = files_diff(
        list_files(ranging_path),
        required_files,
        OPTIONAL_PERF_FILES)
    if diff:
        log.error("%s has file list mismatch (%s)", ranging_path, diff)
        is_valid = False
    diff = files_diff(list_files(power_path), REQUIRED_POWER_FILES)
    if diff:
        log.error("%s has file list mismatch (%s)", power_path, diff)
        is_valid = False

    # uncomment to measure ranging mode power
    """
    (
        is_valid,
        power_metric_ranging,
        scenario,
        power_efficiency_ranging,
    ) = get_power_metric(
        config, scenario_fixed, ranging_path, is_valid, power_res_ranging
    )
    """
    is_valid, power_metric, scenario, power_efficiency_testing = get_power_metric(
        config, scenario_fixed, testing_path, is_valid, power_res_testing
    )

    if not skip_power_check:
        python_version_major = int(sys.version.split(" ")[0].split(".")[0])
        python_version_minor = int(sys.version.split(" ")[0].split(".")[1])
        assert python_version_major == 3 and python_version_minor >= 7, (
            "Power check " " only " "supports " "Python " "3.7+"
        )
        sys.path.insert(0, os.path.join(submission_checker_dir, "power"))
        from power.power_checker import check as check_power_more

        perf_path = os.path.dirname(power_path)
        check_power_result = check_power_more(perf_path)
        sys.stdout.flush()
        sys.stderr.flush()
        if check_power_result != 0:
            log.error(
                "Power WG power_checker.py did not pass for: %s",
                perf_path)
            is_valid = False

    return is_valid, power_metric, power_efficiency_testing


def files_diff(list1, list2, optional=None):
    """returns a list of files that are missing or added."""
    if not optional:
        optional = []
    optional = optional + ["mlperf_log_trace.json", "results.json", ".gitkeep"]
    return set(list1).symmetric_difference(set(list2)) - set(optional)


def is_system_over_network(division, system_json, path):
    """
    Verify whether the submitted system is over network and whether it is valid
    for the division
    for 'network' division, it is mandatory that the system is over-network
    for 'closed' division, the system must not be over-network
    for 'open' division, the system may be either local or over-network
    """
    is_network_mode_sys_spec_str = system_json.get(SYSTEM_DESC_IS_NETWORK_MODE)
    is_network_system = (
        is_network_mode_sys_spec_str.lower() == "true"
        if is_network_mode_sys_spec_str is not None
        else False
    )
    # verify that the system corresponds the division
    is_valid = True
    expected_state_by_division = {"network": True, "closed": False}
    if division in expected_state_by_division:
        is_valid = expected_state_by_division[division] is is_network_system
    if not is_valid:
        log.error(
            f"{path} incorrect network mode (={is_network_system}) for division '{division}'"
        )
    return is_network_system, is_valid


def check_results_dir(
    config,
    filter_submitter,
    csv,
    debug=False,
    skip_meaningful_fields_emptiness_check=False,
    skip_empty_files_check=False,
    skip_check_power_measure_files=False,
    skip_extra_files_in_root_check=False,
    skip_extra_accuracy_files_check=False,
    scenarios_to_skip=[],
):
    """
    Walk the results directory and do the checking.
    We are called with the cdw at the root of the submission directory.
    level1 division - closed|open|network
    level2 submitter - for example mlperf_org
    level3 - results, systems, measurements, code
    For results the structure from here is:
    results/$system_desc/$benchmark_model/$scenario/performance/run_n
    and
    results/$system_desc/$benchmark_model/$scenario/accuracy
    We first walk into results/$system_desc
        make sure there is a system_desc.json and its good
    Next we walk into the model
        make sure the model is good, make sure all required scenarios are there.
    Next we walk into each scenario
        check the performance directory
        check the accuracy directory
        if all was good, add the result to the results directory
        if there are errors write a None as result so we can report later what
        failed
    """
    head = [
        "Organization",
        "Availability",
        "Division",
        "SystemType",
        "SystemName",
        "Platform",
        "Model",
        "MlperfModel",
        "Scenario",
        "Result",
        "Accuracy",
        "number_of_nodes",
        "host_processor_model_name",
        "host_processors_per_node",
        "host_processor_core_count",
        "accelerator_model_name",
        "accelerators_per_node",
        "Location",
        "framework",
        "operating_system",
        "notes",
        "compliance",
        "errors",
        "version",
        "inferred",
        "has_power",
        "Units",
        "weight_data_types",
    ]
    fmt = ",".join(["{}"] * len(head)) + "\n"
    csv.write(",".join(head) + "\n")
    results = {}
    systems = {}

    def log_result(
        submitter,
        available,
        division,
        system_type,
        system_name,
        system_desc,
        model_name,
        mlperf_model,
        scenario_fixed,
        r,
        acc,
        system_json,
        name,
        compliance,
        errors,
        config,
        inferred=0,
        power_metric=0,
        weight_data_types="fp32",
    ):
        notes = system_json.get("hw_notes", "")
        if system_json.get("sw_notes"):
            notes = notes + ". " if notes else ""
            notes = notes + system_json.get("sw_notes")
        special_unit_dict = {}
        unit_dict = {
            "SingleStream": "Latency (ms)",
            "MultiStream": "Latency (ms)",
            "Offline": "Samples/s",
            "ConstantStream": "Queries/s",
        }
        power_unit_dict = {
            "SingleStream": "millijoules",
            "MultiStream": "millijoules",
            "Offline": "Watts",
            "ConstantStream": "Watts",
        }
        if config.version == "v4.0":
            unit = unit_dict[scenario_fixed]
        else:
            unit = special_unit_dict.get(
                mlperf_model, unit_dict).get(
                scenario_fixed, unit_dict[scenario_fixed])
        power_unit = power_unit_dict[scenario_fixed]

        if (power_metric <= 0) or (
            not get_boolean(system_json.get("system_power_only"))
        ):
            csv.write(
                fmt.format(
                    submitter,
                    available,
                    division,
                    '"' + system_type + '"',
                    '"' + system_name + '"',
                    system_desc,
                    model_name,
                    mlperf_model,
                    scenario_fixed,
                    r,
                    acc,
                    system_json.get("number_of_nodes"),
                    '"' + system_json.get("host_processor_model_name") + '"',
                    system_json.get("host_processors_per_node"),
                    system_json.get("host_processor_core_count"),
                    '"' + system_json.get("accelerator_model_name") + '"',
                    '"' + str(system_json.get("accelerators_per_node")) + '"',
                    name.replace("\\", "/"),
                    '"' + system_json.get("framework", "") + '"',
                    '"' + system_json.get("operating_system", "") + '"',
                    '"' + notes + '"',
                    compliance,
                    errors,
                    config.version,
                    inferred,
                    power_metric > 0,
                    unit,
                    '"' + weight_data_types + '"',
                )
            )

        if power_metric > 0:
            csv.write(
                fmt.format(
                    submitter,
                    available,
                    division,
                    '"' + system_type + '"',
                    '"' + system_name + '"',
                    system_desc,
                    model_name,
                    mlperf_model,
                    scenario_fixed,
                    power_metric,
                    acc,
                    system_json.get("number_of_nodes"),
                    '"' + system_json.get("host_processor_model_name") + '"',
                    system_json.get("host_processors_per_node"),
                    system_json.get("host_processor_core_count"),
                    '"' + system_json.get("accelerator_model_name") + '"',
                    '"' + str(system_json.get("accelerators_per_node")) + '"',
                    name.replace("\\", "/"),
                    '"' + system_json.get("framework", "") + '"',
                    '"' + system_json.get("operating_system", "") + '"',
                    '"' + notes + '"',
                    compliance,
                    errors,
                    config.version,
                    inferred,
                    power_metric > 0,
                    power_unit,
                    '"' + weight_data_types + '"',
                )
            )

    # we are at the top of the submission directory
    for division in list_dir("."):
        # we are looking at ./$division, ie ./closed
        if division not in VALID_DIVISIONS:
            if division not in [".git", ".github", "assets"]:
                log.error("invalid division in input dir %s", division)
            continue
        is_closed_or_network = division in ["closed", "network"]
        # Look at files outside the division folder
        files_outside_division = [
            f for f in list_files(".") if not (f.endswith(".md") or f.endswith(".pdf"))
        ]
        if len(files_outside_division) > 0 and not skip_extra_files_in_root_check:
            log.error(
                "Root contains files outside division folder %s. You can use '--skip-extra-files-in-root-check' to skip this check temporarily",
                division,
                files_outside_division,
            )
            results[f"root"] = None
            break
        # Look at files outside the submitter folder
        files_outside_submitter = list_files(division)
        if len(files_outside_submitter) > 0:
            log.error(
                "%s contains files outside submitter folder %s",
                division,
                files_outside_submitter,
            )
            results[f"{division}"] = None
            continue

        if division not in systems:
            systems[division] = {}
            systems[division]["power"] = {}
            systems[division]["non_power"] = {}

        for submitter in list_dir(division):
            # we are looking at ./$division/$submitter, ie ./closed/mlperf_org
            if filter_submitter and submitter != filter_submitter:
                continue
            results_path = os.path.join(division, submitter, "results")
            if not os.path.exists(results_path):
                continue

            # Apply folder checks
            dirs = list_dirs_recursively(division, submitter)
            files = list_files_recursively(division, submitter)

            # Check symbolic links
            broken_symbolic_links = [
                f
                for f in files
                if os.path.islink(f) and not os.path.exists(os.readlink(f))
            ]
            if len(broken_symbolic_links) > 0:
                log.error(
                    "%s/%s contains broken symbolic links: %s",
                    division,
                    submitter,
                    broken_symbolic_links,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check for files over 50 MB
            files_over_size_limit = [
                f
                for f in files
                if os.path.getsize(f) > FILE_SIZE_LIMIT_MB * MB_TO_BYTES
            ]
            if len(files_over_size_limit) > 0:
                log.error(
                    "%s/%s contains files with size greater than 50 MB: %s",
                    division,
                    submitter,
                    files_over_size_limit,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check files and folders with git unfriendly names
            dir_names = [(dir_, dir_.split("/")[-1]) for dir_ in dirs]
            file_names = [(file_, file_.split("/")[-1]) for file_ in files]
            git_error_names = [
                name[0] for name in dir_names if name[1].startswith(".")
            ] + [name[0] for name in file_names if name[1].startswith(".")]
            if len(git_error_names) > 0:
                log.error(
                    "%s/%s contains files with git unfriendly name: %s",
                    division,
                    submitter,
                    git_error_names,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check files and folders with spaces names
            space_error_names = [name[0] for name in dir_names if " " in name[1]] + [
                name[0] for name in file_names if " " in name[1]
            ]
            if len(space_error_names) > 0:
                log.error(
                    "%s/%s contains files with spaces in their names: %s",
                    division,
                    submitter,
                    space_error_names,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check for pycache folders
            pycache_dirs = [dir for dir in dirs if dir.endswith("__pycache__")]
            if len(pycache_dirs) > 0:
                log.error(
                    "%s/%s has the following __pycache__ directories: %s",
                    division,
                    submitter,
                    pycache_dirs,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check for empty folders
            empty_dirs = list_empty_dirs_recursively(division, submitter)
            if len(empty_dirs) > 0:
                log.error(
                    "%s/%s has the following empty directories: %s",
                    division,
                    submitter,
                    empty_dirs,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check for extra model mapping
            extra_model_mapping = None
            if division == "open":
                model_mapping_path = (
                    f"{division}/{submitter}/{config.extra_model_benchmark_map}"
                )
                if os.path.exists(model_mapping_path):
                    with open(model_mapping_path) as fp:
                        extra_model_mapping = json.load(fp)

            for system_desc in list_dir(results_path):
                # we are looking at
                # ./$division/$submitter/results/$system_desc, ie
                # ./closed/mlperf_org/results/t4-ort

                #
                # check if system_id is good.
                #
                system_id_json = os.path.join(
                    division, submitter, "systems", system_desc + ".json"
                )
                if not os.path.exists(system_id_json):
                    log.error(
                        "no system_desc for %s/%s/%s", division, submitter, system_desc
                    )
                    results[os.path.join(results_path, system_desc)] = None
                    continue

                name = os.path.join(results_path, system_desc)
                with open(system_id_json) as f:
                    system_json = json.load(f)
                    available = system_json.get("status").lower()
                    if available not in VALID_AVAILABILITIES:
                        log.error(
                            "%s has invalid status (%s)", system_id_json, available
                        )
                        results[name] = None
                        continue
                    system_type = system_json.get("system_type")
                    valid_system_types = ["adas"]

                    if system_type not in valid_system_types:
                        log.error(
                            "%s has invalid system type (%s)",
                            system_id_json,
                            system_type,
                        )
                        results[name] = None
                        continue

                    config.set_type(system_type)
                    if not check_system_desc_id(
                        name,
                        system_json,
                        submitter,
                        division,
                        config.version,
                        skip_meaningful_fields_emptiness_check,
                    ):
                        results[name] = None
                        continue

                #
                # Look at each model
                #
                for model_name in list_dir(results_path, system_desc):
                    # we are looking at ./$division/$submitter/results/$system_desc/$model,
                    #   ie ./closed/mlperf_org/results/t4-ort/bert
                    name = os.path.join(results_path, system_desc, model_name)
                    mlperf_model = config.get_mlperf_model(
                        model_name, extra_model_mapping
                    )

                    if is_closed_or_network and mlperf_model not in config.models:
                        # for closed/network divisions we want the model name to match.
                        # for open division the model_name might be different
                        # than the task
                        log.error(
                            "%s has an invalid model %s for closed/network division",
                            name,
                            model_name,
                        )
                        results[name] = None
                        continue

                    #
                    # Look at each scenario
                    #
                    required_scenarios = config.get_required(mlperf_model)
                    if required_scenarios is None:
                        log.error(
                            "%s has an invalid model %s, system_type=%s",
                            name,
                            mlperf_model,
                            system_type,
                        )
                        results[name] = None
                        continue

                    errors = 0
                    all_scenarios = set(
                        list(required_scenarios)
                        + list(config.get_optional(mlperf_model))
                    )
                    for scenario in list_dir(
                            results_path, system_desc, model_name):
                        # some submissions in v0.5 use lower case scenarios -
                        # map them for now
                        scenario_fixed = SCENARIO_MAPPING.get(
                            scenario, scenario)

                        # Skip scenario for debug purposes
                        if scenario in scenarios_to_skip:
                            continue

                        # we are looking at ./$division/$submitter/results/$system_desc/$model/$scenario,
                        #   ie ./closed/mlperf_org/results/t4-ort/bert/Offline
                        name = os.path.join(
                            results_path, system_desc, model_name, scenario
                        )
                        results[name] = None
                        if is_closed_or_network and scenario_fixed not in all_scenarios:
                            log.error(
                                "%s ignoring scenario %s (neither required nor optional)",
                                name,
                                scenario,
                            )
                            results[name] = None
                            errors += 1
                            continue

                        # check if this submission has power logs
                        power_path = os.path.join(name, "performance", "power")
                        has_power = os.path.exists(power_path)

                        if has_power:
                            log.info("Detected power logs for %s", name)
                            # The power related system_desc_fields are not used by submitters currently.
                            # Turning this check off for now
                            if False and not check_system_desc_id_power(
                                name,
                                system_json,
                                submitter,
                                division,
                                config.version,
                                skip_meaningful_fields_emptiness_check,
                            ):
                                results[name] = None
                                errors += 1
                                continue

                        # check if measurement_dir is good.
                        measurement_dir = os.path.join(
                            division,
                            submitter,
                            "measurements",
                            system_desc,
                            model_name,
                            scenario,
                        )
                        if not os.path.exists(measurement_dir):
                            log.error(
                                "no measurement_dir for %s", measurement_dir)
                            results[measurement_dir] = None
                            errors += 1
                            continue
                        else:
                            measurement_check, weight_data_types = check_measurement_dir(
                                config,
                                measurement_dir,
                                name,
                                system_desc,
                                os.path.join(division, submitter),
                                model_name,
                                scenario,
                                division,
                                has_power,
                                skip_meaningful_fields_emptiness_check,
                                skip_empty_files_check,
                                skip_check_power_measure_files,
                            )
                            if not measurement_check:
                                log.error(
                                    "%s measurement_dir has issues", measurement_dir
                                )
                                results[measurement_dir] = None
                                errors += 1
                                continue

                        # check accuracy
                        accuracy_is_valid = False
                        acc_path = os.path.join(name, "accuracy")
                        if not os.path.exists(
                                os.path.join(acc_path, "accuracy.txt")):
                            log.error(
                                "%s has no accuracy.txt. Generate it with accuracy-imagenet.py or accuracy-coco.py or "
                                "process_accuracy.py",
                                acc_path,
                            )
                            errors += 1
                            continue
                        elif scenario not in scenarios_to_skip:
                            diff = files_diff(
                                list_files(acc_path), REQUIRED_ACC_FILES)
                            if diff:
                                log.error(
                                    "%s has file list mismatch (%s)", acc_path, diff
                                )
                                errors += 1
                                continue
                            accuracy_is_valid, acc = check_accuracy_dir(
                                config,
                                mlperf_model,
                                acc_path,
                                debug or is_closed_or_network,
                            )
                            acc = (
                                json.dumps(acc)
                                .replace(",", " ")
                                .replace('"', "")
                                .replace("{", "")
                                .replace("}", "")
                            ).strip()
                            if not accuracy_is_valid and acc and not is_closed_or_network:
                                if debug:
                                    log.warning(
                                        "%s, accuracy not valid but taken for open",
                                        acc_path,
                                    )
                                accuracy_is_valid = True
                            if not accuracy_is_valid:
                                # a little below we'll not copy this into the
                                # results csv
                                errors += 1
                                log.error("%s, accuracy not valid", acc_path)

                        inferred = 0
                        n = ["run_1"]

                        for i in n:
                            is_valid = True
                            perf_path = os.path.join(name, "performance", i)
                            if not os.path.exists(perf_path):
                                log.error("%s is missing", perf_path)
                                is_valid, r = False, None
                                continue
                            if has_power:
                                required_perf_files = (
                                    REQUIRED_PERF_FILES + REQUIRED_PERF_POWER_FILES
                                )
                            else:
                                required_perf_files = REQUIRED_PERF_FILES
                            diff = files_diff(
                                list_files(perf_path),
                                required_perf_files,
                                OPTIONAL_PERF_FILES,
                            )
                            if diff:
                                log.error(
                                    "%s has file list mismatch (%s)", perf_path, diff
                                )
                                is_valid, r = False, None
                                continue

                            try:
                                is_valid, r, is_inferred = check_performance_dir(
                                    config,
                                    mlperf_model,
                                    perf_path,
                                    scenario_fixed,
                                    division,
                                    system_json
                                )
                                if is_inferred:
                                    inferred = 1
                                    log.info(
                                        "%s has inferred results, qps=%s", perf_path, r
                                    )

                            except Exception as e:
                                log.error(
                                    "%s caused exception in check_performance_dir: %s",
                                    perf_path,
                                    e,
                                )
                                is_valid, r = False, None

                            power_metric = 0
                            if has_power:
                                try:
                                    ranging_path = os.path.join(
                                        name, "performance", "ranging"
                                    )
                                    ranging_r = get_performance_metric(
                                        config,
                                        mlperf_model,
                                        ranging_path,
                                        scenario_fixed,
                                    )
                                except Exception as e:
                                    log.error(
                                        "%s caused exception in check_ranging_dir: %s",
                                        ranging_path,
                                        e,
                                    )
                                    is_valid, ranging_r = False, None

                                try:
                                    (
                                        power_is_valid,
                                        power_metric,
                                        power_efficiency,
                                    ) = check_power_dir(
                                        power_path,
                                        ranging_path,
                                        perf_path,
                                        scenario_fixed,
                                        ranging_r,
                                        r,
                                        config,
                                    )
                                    if not power_is_valid:
                                        is_valid = False
                                        power_metric = 0
                                except Exception as e:
                                    log.error(
                                        "%s caused exception in check_power_dir: %s",
                                        perf_path,
                                        e,
                                    )
                                    is_valid, r, power_metric = False, None, 0

                            if is_valid:
                                results[name] = (
                                    r
                                    if r is None or not has_power
                                    else (
                                        "{:f} "
                                        "with "
                                        "power_metric"
                                        " = {:f} and power_efficiency (samples/J) = {:f}"
                                    ).format(r, power_metric, power_efficiency)
                                )

                                system_id = submitter + "_" + system_desc

                                key = "power" if power_metric > 0 else "non_power"
                                if system_id not in systems[division][key]:
                                    systems[division][key][system_id] = 1
                                else:
                                    systems[division][key][system_id] += 1

                                required_scenarios.discard(scenario_fixed)
                            else:
                                log.error("%s has issues", perf_path)
                                errors += 1
                                results[name] = None

                        if results.get(name):
                            if accuracy_is_valid:
                                log_result(
                                    submitter,
                                    available,
                                    division,
                                    system_type,
                                    system_json.get("system_name"),
                                    system_desc,
                                    model_name,
                                    mlperf_model,
                                    scenario_fixed,
                                    r,
                                    acc,
                                    system_json,
                                    name,
                                    1,
                                    errors,
                                    config,
                                    inferred=inferred,
                                    power_metric=power_metric,
                                    weight_data_types=weight_data_types,
                                )
                            else:
                                results[name] = None
                                log.error(
                                    "%s is OK but accuracy has issues", name)

                    # Discard scenarios that we want to skip
                    for scenario in scenarios_to_skip:
                        required_scenarios.discard(scenario)

                    if len(required_scenarios) > 1:
                        name = os.path.join(
                            results_path, system_desc, model_name)
                        if is_closed_or_network:
                            results[name] = None
                            log.error(
                                "%s does not have all required scenarios, missing %s",
                                name,
                                required_scenarios,
                            )
                        elif debug:
                            log.warning(
                                "%s ignoring missing scenarios in open division (%s)",
                                name,
                                required_scenarios,
                            )

    return results, systems


def check_system_desc_id(
    fname,
    systems_json,
    submitter,
    division,
    version,
    skip_meaningful_fields_emptiness_check,
):
    is_valid = True
    # check all required fields

    required_fields = SYSTEM_DESC_REQUIRED_FIELDS.copy()

    is_network_system, is_network_mode_valid = is_system_over_network(
        division, systems_json, fname
    )
    is_valid &= is_network_mode_valid
    if is_network_system:
        required_fields += SYSTEM_DESC_REQUIRED_FIELDS_NETWORK_MODE

    check_empty_fields = False if skip_meaningful_fields_emptiness_check else True

    for k in required_fields:
        if k not in systems_json:
            is_valid = False
            log.error("%s, field %s is missing", fname, k)
        elif (
            check_empty_fields
            and k in SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS
            and not systems_json[k]
        ):
            is_valid = False
            log.error(
                "%s, field %s requires a meaningful response but is empty", fname, k
            )

    # SYSTEM_DESC_REQUIRED_FIELDS_POWER should be mandatory when a submission has power logs, but since we
    # check power submission in check_results_dir, the information is not available yet at this stage and we do
    # this check later
    all_fields = required_fields + SYSTEM_DESC_REQUIRED_FIELDS_POWER
    for k in systems_json.keys():
        if k not in all_fields:
            log.warning("%s, field %s is unknown", fname, k)

    if systems_json.get("submitter").lower() != submitter.lower():
        log.error(
            "%s has submitter %s, directory has %s",
            fname,
            systems_json.get("submitter"),
            submitter,
        )
        is_valid = False
    if systems_json.get("division") != division:
        log.error(
            "%s has division %s, division has %s",
            fname,
            systems_json.get("division"),
            division,
        )
        is_valid = False
    return is_valid


def check_system_desc_id_power(
    fname,
    systems_json,
    submitter,
    division,
    version,
    skip_meaningful_fields_emptiness_check,
):
    is_valid = True

    check_empty_fields = False if skip_meaningful_fields_emptiness_check else True

    for k in SYSTEM_DESC_REQUIRED_FIELDS_POWER:
        if k not in systems_json:
            is_valid = False
            log.error("%s, field %s is missing", fname, k)
        elif (
            check_empty_fields
            and k in SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS_POWER
            and not systems_json[k]
        ):
            is_valid = False
            log.error(
                "%s, field %s requires a meaningful response but is empty", fname, k
            )

    return is_valid


def check_measurement_dir(
    config,
    measurement_dir,
    fname,
    system_desc,
    root,
    model,
    scenario,
    division,
    has_power,
    skip_meaningful_fields_emptiness_check,
    skip_empty_files_check,
    skip_check_power_measure_files,
):
    files = list_files(measurement_dir)
    system_file = None
    is_valid = True
    check_empty_fields = False if skip_meaningful_fields_emptiness_check else True

    for i in REQUIRED_MEASURE_FILES:
        if i not in files:
            log.error("%s is missing %s", measurement_dir, i)
            is_valid = False
        elif not skip_empty_files_check and (
            os.stat(os.path.join(measurement_dir, i)).st_size == 0
        ):
            log.error("%s is having empty %s", measurement_dir, i)
            is_valid = False

    if has_power and not skip_check_power_measure_files:
        path = measurement_dir
        all_files_1 = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        path = os.path.join(path, "..")
        all_files_2 = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        path = os.path.join(path, "..")
        all_files_3 = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        path = os.path.join(path, "..")
        all_files_4 = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        all_files = all_files_1 + all_files_2 + all_files_3 + all_files_4

        for i in REQUIRED_POWER_MEASURE_FILES:
            found = False
            for file in all_files:
                if re.match(i, os.path.basename(file)):
                    found = True
                    file_path = file
            if not found:
                log.error("%s is missing %s", measurement_dir, i)
                is_valid = False
            elif not skip_empty_files_check and os.stat(file_path).st_size == 0:
                log.error("%s is having empty %s", measurement_dir, i)
                is_valid = False

    if config.version in ["v4.0", "v4.1"]:
        system_file_prefix = system_desc
    else:
        system_file_prefix = "model-info"
    for i in files:
        if i.startswith(system_desc) and i.endswith(
                "_" + scenario + ".json"):
            system_file = i
            end = len("_" + scenario + ".json")
            break
        elif i.startswith(system_desc) and i.endswith(".json"):
            system_file = i
            end = len(".json")
            break

    weight_data_types = None
    if system_file:
        with open(os.path.join(measurement_dir, system_file), "r") as f:
            j = json.load(f)
            weight_data_types = j["weight_data_types"]
            for k in SYSTEM_IMP_REQUIRED_FILES:
                if k not in j:
                    is_valid = False
                    log.error("%s, field %s is missing", fname, k)
                elif check_empty_fields and not j[k]:
                    is_valid = False
                    log.error(
                        "%s, field %s is missing meaningful value", fname, k)

        impl = system_file[len(system_desc) + 1: -end]
        code_dir = os.path.join(root, "code", model)
        if os.path.isfile(code_dir):
            with open(code_dir, "r") as f:
                line = f.read()
                code_dir = os.path.join(root, "code", line.strip(), impl)
        else:
            code_dir = os.path.join(root, "code", model, impl)

        if not os.path.exists(code_dir):
            # see if the code dir is per model
            if not os.path.exists(os.path.dirname(code_dir)):
                log.error("%s is missing code_dir %s", fname, code_dir)
                is_valid = False

    else:
        log.error("%s is missing %s*.json", fname, system_desc)
        is_valid = False

    return is_valid, weight_data_types


def main():
    args = get_args()

    config = Config(
        args.version,
        args.extra_model_benchmark_map,
        ignore_uncommited=args.submission_exceptions,
        skip_power_check=args.skip_power_check,
    )

    if args.scenarios_to_skip:
        scenarios_to_skip = [
            scenario for scenario in args.scenarios_to_skip.split(",")]
    else:
        scenarios_to_skip = []

    with open(args.csv, "w") as csv:
        os.chdir(args.input)
        # check results directory
        results, systems = check_results_dir(
            config,
            args.submitter,
            csv,
            args.debug,
            args.skip_meaningful_fields_emptiness_check,
            args.skip_empty_files_check,
            args.skip_check_power_measure_files,
            args.skip_extra_files_in_root_check,
            args.skip_extra_accuracy_files_check,
            scenarios_to_skip,
        )

    # log results
    log.info("---")
    with_results = 0
    for k, v in sorted(results.items()):
        if v:
            log.info("Results %s %s", k, v)
            with_results += 1
    log.info("---")
    for k, v in sorted(results.items()):
        if v is None:
            log.error("NoResults %s", k)

    closed_systems = systems.get("closed", {})
    open_systems = systems.get("open", {})
    network_systems = systems.get("network", {})
    closed_power_systems = closed_systems.get("power", {})
    closed_non_power_systems = closed_systems.get("non_power", {})
    open_power_systems = open_systems.get("power", {})
    open_non_power_systems = open_systems.get("non_power", {})
    network_power_systems = network_systems.get("power", {})
    network_non_power_systems = network_systems.get("non_power", {})

    number_closed_power_systems = len(closed_power_systems)
    number_closed_non_power_systems = len(closed_non_power_systems)
    number_closed_systems = (
        number_closed_power_systems + number_closed_non_power_systems
    )
    number_open_power_systems = len(open_power_systems)
    number_open_non_power_systems = len(open_non_power_systems)
    number_open_systems = number_open_power_systems + number_open_non_power_systems
    number_network_power_systems = len(network_power_systems)
    number_network_non_power_systems = len(network_non_power_systems)
    number_network_systems = (
        number_network_power_systems + number_network_non_power_systems
    )

    def merge_two_dict(x, y):
        z = x.copy()
        for key in y:
            if key not in z:
                z[key] = y[key]
            else:
                z[key] += y[key]
        return z

    # systems can be repeating in open, closed and network
    unique_closed_systems = merge_two_dict(
        closed_power_systems, closed_non_power_systems
    )
    unique_open_systems = merge_two_dict(
        open_power_systems, open_non_power_systems)
    unique_network_systems = merge_two_dict(
        network_power_systems, network_non_power_systems
    )

    unique_systems = merge_two_dict(unique_closed_systems, unique_open_systems)
    unique_systems = merge_two_dict(unique_systems, unique_network_systems)

    # power systems can be repeating in open, closed and network
    unique_power_systems = merge_two_dict(
        closed_power_systems, open_power_systems)
    unique_power_systems = merge_two_dict(
        unique_power_systems, network_power_systems)

    number_systems = len(unique_systems)
    number_power_systems = len(unique_power_systems)

    # Counting the number of closed,open and network results
    def sum_dict_values(x):
        count = 0
        for key in x:
            count += x[key]
        return count

    count_closed_power_results = sum_dict_values(closed_power_systems)
    count_closed_non_power_results = sum_dict_values(closed_non_power_systems)
    count_closed_results = count_closed_power_results + count_closed_non_power_results

    count_open_power_results = sum_dict_values(open_power_systems)
    count_open_non_power_results = sum_dict_values(open_non_power_systems)
    count_open_results = count_open_power_results + count_open_non_power_results

    count_network_power_results = sum_dict_values(network_power_systems)
    count_network_non_power_results = sum_dict_values(
        network_non_power_systems)
    count_network_results = (
        count_network_power_results + count_network_non_power_results
    )

    count_power_results = (
        count_closed_power_results
        + count_open_power_results
        + count_network_power_results
    )

    # print summary
    log.info("---")
    log.info(
        "Results=%d, NoResults=%d, Power Results=%d",
        with_results,
        len(results) - with_results,
        count_power_results,
    )

    log.info("---")
    log.info(
        "Closed Results=%d, Closed Power Results=%d\n",
        count_closed_results,
        count_closed_power_results,
    )
    log.info(
        "Open Results=%d, Open Power Results=%d\n",
        count_open_results,
        count_open_power_results,
    )
    log.info(
        "Network Results=%d, Network Power Results=%d\n",
        count_network_results,
        count_network_power_results,
    )
    log.info("---")

    log.info(
        "Systems=%d, Power Systems=%d",
        number_systems,
        number_power_systems)
    log.info(
        "Closed Systems=%d, Closed Power Systems=%d",
        number_closed_systems,
        number_closed_power_systems,
    )
    log.info(
        "Open Systems=%d, Open Power Systems=%d",
        number_open_systems,
        number_open_power_systems,
    )
    log.info(
        "Network Systems=%d, Network Power Systems=%d",
        number_network_systems,
        number_network_power_systems,
    )
    log.info("---")
    if len(results) != with_results:
        log.error("SUMMARY: submission has errors")
        return 1
    else:
        log.info("SUMMARY: submission looks OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
