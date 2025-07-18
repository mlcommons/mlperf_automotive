/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "test_settings_internal.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "logging.h"
#include "mlperf_conf.h"
#include "utils.h"

namespace mlperf {
namespace loadgen {

TestSettingsInternal::TestSettingsInternal(
    const TestSettings &requested_settings, size_t qsl_performance_sample_count,
    size_t qsl_total_sample_count)
    : requested(requested_settings),
      scenario(requested.scenario),
      mode(requested.mode),
      samples_per_query(1),
      target_qps(1),
      max_async_queries(0),
      target_duration(std::chrono::milliseconds(requested.min_duration_ms)),
      min_duration(std::chrono::milliseconds(requested.min_duration_ms)),
      max_duration(std::chrono::milliseconds(requested.max_duration_ms)),
      min_query_count(requested.min_query_count),
      max_query_count(requested.max_query_count),
      min_sample_count(0),
      qsl_rng_seed(requested.qsl_rng_seed),
      sample_index_rng_seed(requested.sample_index_rng_seed),
      schedule_rng_seed(requested.schedule_rng_seed),
      accuracy_log_rng_seed(requested.accuracy_log_rng_seed),
      accuracy_log_probability(requested.accuracy_log_probability),
      accuracy_log_sampling_target(requested.accuracy_log_sampling_target),
      print_timestamps(requested.print_timestamps),
      performance_issue_unique(requested.performance_issue_unique),
      performance_issue_same(requested.performance_issue_same),
      performance_issue_same_index(requested.performance_issue_same_index),
      performance_sample_count(0),
      sample_concatenate_permutation(false),
      use_token_latencies(requested.use_token_latencies),
      server_ttft_latency(requested.server_ttft_latency),
      server_tpot_latency(requested.server_tpot_latency),
      server_constant_gen(requested.server_constant_gen),
      infer_token_latencies(requested.infer_token_latencies),
      token_latency_scaling_factor(requested.token_latency_scaling_factor),
      use_grouped_qsl(requested.use_grouped_qsl),
      group_sizes(requested.group_sizes) {
  // Target QPS, target latency, and max_async_queries.
  switch (requested.scenario) {
    case TestScenario::SingleStream:
      target_qps = static_cast<double>(std::nano::den) /
                   requested.single_stream_expected_latency_ns;
      max_async_queries = 1;
      target_latency_percentile =
          requested.single_stream_target_latency_percentile;
      break;
    case TestScenario::MultiStream:
      target_qps = static_cast<double>(std::nano::den) /
                   requested.multi_stream_expected_latency_ns;
      max_async_queries = 1;
      target_latency_percentile =
          requested.multi_stream_target_latency_percentile;
      break;
    case TestScenario::ConstantStream:{
      if (requested.server_target_qps >= 0.0) {
        target_qps = requested.server_target_qps;
      } else {
        LogDetail([server_target_qps = requested.server_target_qps,
                   target_qps = target_qps](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
          std::stringstream ss;
          ss << "Invalid value for server_target_qps requested."
             << " requested: " << server_target_qps << " using: " << target_qps;
          MLPERF_LOG_ERROR(detail, "error_invalid_test_settings", ss.str());
#else
          detail.Error("Invalid value for server_target_qps requested.",
                       "requested", server_target_qps, "using", target_qps);
#endif
        });
      }
      target_latency = std::chrono::nanoseconds(uint64_t(1000000000 / target_qps));
      target_latency_percentile = requested.server_target_latency_percentile;
      max_async_queries = requested.server_max_async_queries;
      break;
    }
    case TestScenario::Offline:
      // target_latency_percentile is not used in Offline, but set it to
      // 0.99 anyway to avoid garbage value.
      target_latency_percentile = 0.99;
      if (requested.offline_expected_qps >= 0.0) {
        target_qps = requested.offline_expected_qps;
      } else {
        LogDetail([offline_expected_qps = requested.offline_expected_qps,
                   target_qps = target_qps](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
          std::stringstream ss;
          ss << "Invalid value for offline_expected_qps requested."
             << " requested: " << offline_expected_qps
             << " using: " << target_qps;
          MLPERF_LOG_ERROR(detail, "error_invalid_test_settings", ss.str());
#else
          detail.Error("Invalid value for offline_expected_qps requested.",
                       "requested", offline_expected_qps, "using", target_qps);
#endif
        });
      }
      max_async_queries = 1;
      break;
  }

  if (use_grouped_qsl && group_sizes.empty()) {
    for (size_t i = 0; i < qsl_total_sample_count; i++) {
      group_sizes.push_back(1);
    }
  }

  // Performance Sample Count: TestSettings override QSL ->
  // PerformanceSampleCount
  performance_sample_count = (requested.performance_sample_count_override == 0)
                                 ? qsl_performance_sample_count
                                 : requested.performance_sample_count_override;

  // Sample by concatentating several permutations of the dataset
  // sample_concatenate_permutation
  sample_concatenate_permutation =
      (requested.sample_concatenate_permutation == 0)
          ? false
          : requested.sample_concatenate_permutation;

  // Samples per query.
  if (requested.scenario == TestScenario::MultiStream) {
    samples_per_query = requested.multi_stream_samples_per_query;
  }

  // In the offline scenario, coalesce all queries into a single query.
  if (requested.scenario == TestScenario::Offline) {
    // TODO: Should the spec require a max duration for large query counts?
    // kSlack is used to make sure we generate enough samples for the SUT
    // to take longer than than the minimum test duration required by the
    // MLPerf spec.
    constexpr double kSlack = 1.1;
    uint64_t target_sample_count =
        kSlack * DurationToSeconds(target_duration) * target_qps;
    samples_per_query =
        (requested.performance_issue_unique)
            ? performance_sample_count
            : std::max<uint64_t>(min_query_count, target_sample_count);
    min_query_count = 1;
    target_duration = std::chrono::milliseconds(0);
  }

  // FIXME: Only do this for 3D-UNet SingleStream, for v2.0
  // TODO: consolidate after v2.0
  // make min_queries to be multiple of performance_sample_count
  // performance_sample_count == 0 makes it to be equal to loaded_samples.size()
  if (sample_concatenate_permutation &&
      requested.scenario == TestScenario::SingleStream) {
    // set slack larger for 3D-UNet KiTS19 distribution, i.e. 50% latency << 90%
    // latency
    constexpr double kSlack = 2.0;
    uint64_t expected_queries =
        kSlack * DurationToSeconds(target_duration) * target_qps;
    min_query_count =
        min_query_count > expected_queries ? min_query_count : expected_queries;
    min_query_count += qsl_performance_sample_count -
                       (min_query_count % qsl_performance_sample_count);
  }

  min_sample_count = min_query_count * samples_per_query;

  // Validate TestSettings
  if (requested.performance_issue_same &&
      (requested.performance_issue_same_index >= performance_sample_count)) {
    LogDetail([performance_issue_same_index =
                   requested.performance_issue_same_index,
               performance_sample_count =
                   performance_sample_count](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
      std::stringstream ss;
      ss << "Sample Idx to be repeated in performance_issue_same mode"
         << " cannot be greater than loaded performance_sample_count."
         << " performance_issue_same_index: " << performance_issue_same_index
         << " performance_sample_count: " << performance_sample_count;
      MLPERF_LOG_ERROR(detail, "error_invalid_test_settings", ss.str());
#else
      detail.Error(
          "Sample Idx to be repeated in performance_issue_same mode"
          " cannot be greater than loaded performance_sample_count.",
          "performance_issue_same_index", performance_issue_same_index,
          "performance_sample_count", performance_sample_count);
#endif
    });
  }

  if (requested.performance_issue_unique && requested.performance_issue_same) {
    LogDetail([performance_issue_unique = requested.performance_issue_unique,
               performance_issue_same =
                   requested.performance_issue_same](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
      std::stringstream ss;
      ss << "Performance_issue_unique and performance_issue_same, both"
         << " cannot be true at the same time."
         << " performance_issue_unique: " << performance_issue_unique
         << " performance_issue_same: " << performance_issue_same;
      MLPERF_LOG_ERROR(detail, "error_invalid_test_settings", ss.str());
#else
      detail.Error(
          "Performance_issue_unique and performance_issue_same, both"
          " cannot be true at the same time.",
          "performance_issue_unique", performance_issue_unique,
          "performance_issue_same", performance_issue_same);
#endif
    });
  }
}

std::string ToString(TestScenario scenario) {
  switch (scenario) {
#if USE_NEW_LOGGING_FORMAT
    case TestScenario::SingleStream:
      return "SingleStream";
    case TestScenario::MultiStream:
      return "MultiStream";
#else
    case TestScenario::SingleStream:
      return "Single Stream";
    case TestScenario::MultiStream:
      return "Multi Stream";
#endif
    case TestScenario::ConstantStream:
      return "ConstantStream";
    case TestScenario::Offline:
      return "Offline";
  }
  assert(false);
  return "InvalidScenario";
}

std::string ToString(TestMode mode) {
  switch (mode) {
#if USE_NEW_LOGGING_FORMAT
    case TestMode::SubmissionRun:
      return "SubmissionRun";
    case TestMode::AccuracyOnly:
      return "AccuracyOnly";
    case TestMode::PerformanceOnly:
      return "PerformanceOnly";
    case TestMode::FindPeakPerformance:
      return "FindPeakPerformance";
#else
    case TestMode::SubmissionRun:
      return "Submission";
    case TestMode::AccuracyOnly:
      return "Accuracy";
    case TestMode::PerformanceOnly:
      return "Performance";
    case TestMode::FindPeakPerformance:
      return "Find Peak Performance";
#endif
  }
  assert(false);
  return "InvalidMode";
}

void LogRequestedTestSettings(const TestSettings &s) {
  LogDetail([s](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "requested_scenario", ToString(s.scenario));
    MLPERF_LOG(detail, "requested_test_mode", ToString(s.mode));

    // Scenario-specific
    switch (s.scenario) {
      case TestScenario::SingleStream:
        MLPERF_LOG(detail, "requested_single_stream_expected_latency_ns",
                   s.single_stream_expected_latency_ns);
        MLPERF_LOG(detail, "requested_single_stream_target_latency_percentile",
                   s.single_stream_target_latency_percentile);
        break;
      case TestScenario::MultiStream:
        MLPERF_LOG(detail, "requested_multi_stream_expected_latency_ns",
                   s.multi_stream_expected_latency_ns);
        MLPERF_LOG(detail, "requested_multi_stream_target_latency_percentile",
                   s.multi_stream_target_latency_percentile);
        MLPERF_LOG(detail, "requested_multi_stream_samples_per_query",
                   s.multi_stream_samples_per_query);
        break;
      case TestScenario::ConstantStream:
        MLPERF_LOG(detail, "requested_server_target_qps", s.server_target_qps);
        MLPERF_LOG(detail, "requested_server_target_latency_ns",
                   s.server_target_latency_ns);
        MLPERF_LOG(detail, "requested_server_target_latency_percentile",
                   s.server_target_latency_percentile);
        MLPERF_LOG(detail, "requested_server_coalesce_queries",
                   s.server_coalesce_queries);
        MLPERF_LOG(detail,
                   "requested_server_find_peak_qps_decimals_of_precision",
                   s.server_find_peak_qps_decimals_of_precision);
        MLPERF_LOG(detail, "requested_server_find_peak_qps_boundary_step_size",
                   s.server_find_peak_qps_boundary_step_size);
        MLPERF_LOG(detail, "requested_server_max_async_queries",
                   s.server_max_async_queries);
        MLPERF_LOG(detail, "requested_server_num_issue_query_threads",
                   s.server_num_issue_query_threads);
        MLPERF_LOG(detail, "requested_server_constant_gen",
                   s.server_constant_gen);
        break;
      case TestScenario::Offline:
        MLPERF_LOG(detail, "requested_offline_expected_qps",
                   s.offline_expected_qps);
        break;
    }

    // Overrides
    MLPERF_LOG(detail, "requested_min_duration_ms", s.min_duration_ms);
    MLPERF_LOG(detail, "requested_max_duration_ms", s.max_duration_ms);
    MLPERF_LOG(detail, "requested_min_query_count", s.min_query_count);
    MLPERF_LOG(detail, "requested_max_query_count", s.max_query_count);
    MLPERF_LOG(detail, "requested_qsl_rng_seed", s.qsl_rng_seed);
    MLPERF_LOG(detail, "requested_sample_index_rng_seed",
               s.sample_index_rng_seed);
    MLPERF_LOG(detail, "requested_schedule_rng_seed", s.schedule_rng_seed);
    MLPERF_LOG(detail, "requested_accuracy_log_rng_seed",
               s.accuracy_log_rng_seed);
    MLPERF_LOG(detail, "requested_accuracy_log_probability",
               s.accuracy_log_probability);
    MLPERF_LOG(detail, "requested_accuracy_log_sampling_target",
               s.accuracy_log_sampling_target);
    MLPERF_LOG(detail, "requested_print_timestamps", s.print_timestamps);
    MLPERF_LOG(detail, "requested_performance_issue_unique",
               s.performance_issue_unique);
    MLPERF_LOG(detail, "requested_performance_issue_same",
               s.performance_issue_same);
    MLPERF_LOG(detail, "requested_performance_issue_same_index",
               s.performance_issue_same_index);
    MLPERF_LOG(detail, "requested_performance_sample_count_override",
               s.performance_sample_count_override);
    MLPERF_LOG(detail, "requested_sample_concatenate_permutation",
               s.sample_concatenate_permutation);
    MLPERF_LOG(detail, "requested_server_constant_gen", s.server_constant_gen);
    MLPERF_LOG(detail, "requested_use_grouped_qsl", s.use_grouped_qsl);
    // Token latencies specific values
    if (s.use_token_latencies) {
      MLPERF_LOG(detail, "requested_use_token_latencies",
                 s.use_token_latencies);
      if (s.scenario != TestScenario::Offline) {
        MLPERF_LOG(detail, "requested_server_ttft_latency",
                   s.server_ttft_latency);
        MLPERF_LOG(detail, "requested_server_tpot_latency",
                   s.server_tpot_latency);
      }
    }
#else
    detail("");
    detail("Requested Settings:");
    detail("Scenario : " + ToString(s.scenario));
    detail("Test mode : " + ToString(s.mode));

    // Scenario-specific
    switch (s.scenario) {
      case TestScenario::SingleStream:
        detail("single_stream_expected_latency_ns : ",
               s.single_stream_expected_latency_ns);
        detail("single_stream_target_latency_percentile : ",
               s.single_stream_target_latency_percentile);
        break;
      case TestScenario::MultiStream:
        detail("multi_stream_expected_latency_ns : ",
               s.multi_stream_expected_latency_ns);
        detail("multi_stream_target_latency_percentile : ",
               s.multi_stream_target_latency_percentile);
        detail("multi_stream_samples_per_query : ",
               s.multi_stream_samples_per_query);
        break;
      case TestScenario::ConstantStream:
        detail("server_target_qps : ", s.server_target_qps);
        detail("server_target_latency_ns : ", s.server_target_latency_ns);
        detail("server_target_latency_percentile : ",
               s.server_target_latency_percentile);
        detail("server_coalesce_queries : ", s.server_coalesce_queries);
        detail("server_find_peak_qps_decimals_of_precision : ",
               s.server_find_peak_qps_decimals_of_precision);
        detail("server_find_peak_qps_boundary_step_size : ",
               s.server_find_peak_qps_boundary_step_size);
        detail("server_max_async_queries : ", s.server_max_async_queries);
        detail("server_num_issue_query_threads : ",
               s.server_num_issue_query_threads);
        break;
      case TestScenario::Offline:
        detail("offline_expected_qps : ", s.offline_expected_qps);
        break;
    }

    // Overrides
    detail("min_duration_ms : ", s.min_duration_ms);
    detail("max_duration_ms : ", s.max_duration_ms);
    detail("min_query_count : ", s.min_query_count);
    detail("max_query_count : ", s.max_query_count);
    detail("qsl_rng_seed : ", s.qsl_rng_seed);
    detail("sample_index_rng_seed : ", s.sample_index_rng_seed);
    detail("schedule_rng_seed : ", s.schedule_rng_seed);
    detail("accuracy_log_rng_seed : ", s.accuracy_log_rng_seed);
    detail("accuracy_log_probability : ", s.accuracy_log_probability);
    detail("accuracy_log_sampling_target : ", s.accuracy_log_sampling_target);
    detail("print_timestamps : ", s.print_timestamps);
    detail("performance_issue_unique : ", s.performance_issue_unique);
    detail("performance_issue_same : ", s.performance_issue_same);
    detail("performance_issue_same_index : ", s.performance_issue_same_index);
    detail("performance_sample_count_override : ",
           s.performance_sample_count_override);
    detail("");
#endif
  });
}

void TestSettingsInternal::LogEffectiveSettings() const {
  LogDetail([s = *this](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "effective_scenario", ToString(s.scenario));
    MLPERF_LOG(detail, "effective_test_mode", ToString(s.mode));

    MLPERF_LOG(detail, "effective_samples_per_query", s.samples_per_query);
    MLPERF_LOG(detail, "effective_target_qps", s.target_qps);
    MLPERF_LOG(detail, "effective_target_latency_ns", s.target_latency.count());
    MLPERF_LOG(detail, "effective_target_latency_percentile",
               s.target_latency_percentile);
    MLPERF_LOG(detail, "effective_max_async_queries", s.max_async_queries);
    MLPERF_LOG(detail, "effective_target_duration_ms",
               s.target_duration.count());
    MLPERF_LOG(detail, "effective_min_duration_ms", s.min_duration.count());
    MLPERF_LOG(detail, "effective_max_duration_ms", s.max_duration.count());
    MLPERF_LOG(detail, "effective_min_query_count", s.min_query_count);
    MLPERF_LOG(detail, "effective_max_query_count", s.max_query_count);
    MLPERF_LOG(detail, "effective_min_sample_count", s.min_sample_count);
    MLPERF_LOG(detail, "effective_qsl_rng_seed", s.qsl_rng_seed);
    MLPERF_LOG(detail, "effective_sample_index_rng_seed",
               s.sample_index_rng_seed);
    MLPERF_LOG(detail, "effective_schedule_rng_seed", s.schedule_rng_seed);
    MLPERF_LOG(detail, "effective_accuracy_log_rng_seed",
               s.accuracy_log_rng_seed);
    MLPERF_LOG(detail, "effective_accuracy_log_probability",
               s.accuracy_log_probability);
    MLPERF_LOG(detail, "effective_accuracy_log_sampling_target",
               s.accuracy_log_sampling_target);
    MLPERF_LOG(detail, "effective_print_timestamps", s.print_timestamps);
    MLPERF_LOG(detail, "effective_performance_issue_unique",
               s.performance_issue_unique);
    MLPERF_LOG(detail, "effective_performance_issue_same",
               s.performance_issue_same);
    MLPERF_LOG(detail, "effective_performance_issue_same_index",
               s.performance_issue_same_index);
    MLPERF_LOG(detail, "effective_performance_sample_count",
               s.performance_sample_count);
    MLPERF_LOG(detail, "effective_sample_concatenate_permutation",
               s.sample_concatenate_permutation);
    MLPERF_LOG(detail, "effective_server_constant_gen", s.server_constant_gen);
    MLPERF_LOG(detail, "effective_use_grouped_qsl", s.use_grouped_qsl);
#else
    detail("");
    detail("Effective Settings:");

    detail("Scenario : " + ToString(s.scenario));
    detail("Test mode : " + ToString(s.mode));

    detail("samples_per_query : ", s.samples_per_query);
    detail("target_qps : ", s.target_qps);
    detail("target_latency (ns): ", s.target_latency.count());
    detail("target_latency_percentile : ", s.target_latency_percentile);
    detail("max_async_queries : ", s.max_async_queries);
    detail("target_duration (ms): ", s.target_duration.count());
    detail("min_duration (ms): ", s.min_duration.count());
    detail("max_duration (ms): ", s.max_duration.count());
    detail("min_query_count : ", s.min_query_count);
    detail("max_query_count : ", s.max_query_count);
    detail("min_sample_count : ", s.min_sample_count);
    detail("qsl_rng_seed : ", s.qsl_rng_seed);
    detail("sample_index_rng_seed : ", s.sample_index_rng_seed);
    detail("schedule_rng_seed : ", s.schedule_rng_seed);
    detail("accuracy_log_rng_seed : ", s.accuracy_log_rng_seed);
    detail("accuracy_log_probability : ", s.accuracy_log_probability);
    detail("accuracy_log_sampling_target : ", s.accuracy_log_sampling_target);
    detail("print_timestamps : ", s.print_timestamps);
    detail("performance_issue_unique : ", s.performance_issue_unique);
    detail("performance_issue_same : ", s.performance_issue_same);
    detail("performance_issue_same_index : ", s.performance_issue_same_index);
    detail("performance_sample_count : ", s.performance_sample_count);
#endif
  });
}

void TestSettingsInternal::LogAllSettings() const {
  LogRequestedTestSettings(requested);
  LogEffectiveSettings();
}

void TestSettingsInternal::LogSummary(AsyncSummary &summary) const {
  summary("samples_per_query : ", samples_per_query);
  summary("target_qps : ", target_qps);
  if (!use_token_latencies) {
    summary("target_latency (ns): ", target_latency.count());
  } else {
    summary("ttft_latency (ns): ", server_ttft_latency);
    summary("tpot_latency (ns): ", server_tpot_latency);
  }
  summary("target_latency_percentile : ", target_latency_percentile);
  summary("max_async_queries : ", max_async_queries);
  summary("min_duration (ms): ", min_duration.count());
  summary("max_duration (ms): ", max_duration.count());
  summary("min_query_count : ", min_query_count);
  summary("max_query_count : ", max_query_count);
  summary("qsl_rng_seed : ", qsl_rng_seed);
  summary("sample_index_rng_seed : ", sample_index_rng_seed);
  summary("schedule_rng_seed : ", schedule_rng_seed);
  summary("accuracy_log_rng_seed : ", accuracy_log_rng_seed);
  summary("accuracy_log_probability : ", accuracy_log_probability);
  summary("accuracy_log_sampling_target : ", accuracy_log_sampling_target);
  summary("print_timestamps : ", print_timestamps);
  summary("performance_issue_unique : ", performance_issue_unique);
  summary("performance_issue_same : ", performance_issue_same);
  summary("performance_issue_same_index : ", performance_issue_same_index);
  summary("performance_sample_count : ", performance_sample_count);
  if (sample_concatenate_permutation) {
    summary(
        "WARNING: sample_concatenate_permutation was set to true. \n"
        "Generated samples per query might be different as the one in the "
        "setting.\n"
        "Check the generated_samples_per_query line in the detailed log for "
        "the real\n"
        "samples_per_query value");
  }
}
}  // namespace loadgen

int TestSettings::FromConfig(const std::string &path, const std::string &model,
                             const std::string &scenario, int conf_type) {
  std::map<std::string, std::string> kv;
  static int configCount = 0;

  if (conf_type == 1) {
    if (configCount == 0) {
      // Only allow userConf as the single configFile and loadgen loads the
      // mlperfConf automatically for perf and accuracy runs
      FromConfig("", model, scenario, 0);
    }

    else {
      LogDetail([](AsyncDetail &detail) {
        std::stringstream ss;
        ss << "Multiple conf files are used. This is not valid for official "
              "submission.";
        MLPERF_LOG_ERROR(detail, "error_invalid_config", ss.str());
      });
    }
    configCount++;
  }

  // lookup key/value pairs from config
  auto lookupkv = [&](const std::string &model, const std::string &scenario,
                      const std::string &key, uint64_t *val_l, double *val_d,
                      double multiplier = 1.0) {
    std::map<std::string, std::string>::iterator it;
    std::string found;
    // lookup exact key first
    it = kv.find(model + "." + scenario + "." + key);
    if (it != kv.end()) {
      found = it->second;
    } else {
      // lookup key with model wildcard
      it = kv.find("*." + scenario + "." + key);
      if (it != kv.end()) {
        found = it->second;
      } else {
        it = kv.find(model + ".*." + key);
        if (it != kv.end()) {
          found = it->second;
        } else {
          it = kv.find("*.*." + key);
          if (it != kv.end()) {
            found = it->second;
          } else {
            return false;
          }
        }
      }
    }
    // if we get here, found will be set
    if (val_l) {
      *val_l = strtoull(found.c_str(), nullptr, 0) *
               static_cast<uint64_t>(multiplier);
    }
    if (val_d) *val_d = strtod(found.c_str(), nullptr) * multiplier;
    return true;
  };

  auto lookupkvstr = [&](const std::string &model, const std::string &scenario,
                         const std::string &key, std::string *val_str) {
    std::map<std::string, std::string>::iterator it;
    std::string found;
    // lookup exact key first
    it = kv.find(model + "." + scenario + "." + key);
    if (it != kv.end()) {
      found = it->second;
    } else {
      // lookup key with model wildcard
      it = kv.find("*." + scenario + "." + key);
      if (it != kv.end()) {
        found = it->second;
      } else {
        it = kv.find(model + ".*." + key);
        if (it != kv.end()) {
          found = it->second;
        } else {
          it = kv.find("*.*." + key);
          if (it != kv.end()) {
            found = it->second;
          } else {
            return false;
          }
        }
      }
    }
    // if we get here, found will be set
    if (val_str) {
      *val_str = found.c_str();
    }
    return true;
  };

  int line_nr = 0;
  int errors = 0;
  // Declare the input stream before the if-else block
  std::unique_ptr<std::istream> fss;
  std::string line;

  if (conf_type != 0) {
    // dirt simple config parser
    fss = std::make_unique<std::ifstream>(path);
    if (!static_cast<std::ifstream *>(fss.get())->is_open()) {
      LogDetail([p = path](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
        std::stringstream ss;
        ss << "can't open file " << p;
        MLPERF_LOG_ERROR(detail, "error_invalid_config", ss.str());
#else
        detail.Error("can't open file ", p);
#endif
      });
      return -ENOENT;
    }
  } else {
    // Convert unsigned char array to std::string
    std::string config_str(mlperf_conf);
    fss = std::make_unique<std::istringstream>(config_str);
  }
  while (std::getline(*fss, line)) {
    line_nr++;
    std::istringstream iss(line);
    std::string s, k;
    int looking_for = 0;  // 0=key, 1=equal, 2=value
    while (iss >> s) {
      if (s == "#" && looking_for != 2) {
        // done with this line
        break;
      }
      if (looking_for == 2) {
        // got key and value
        const char *start = s.c_str();
        char *stop;
        (void)strtoul(start, &stop, 0);
        if (start + s.size() == stop) {
          kv[k] = s;
          continue;
        }
        (void)strtod(start, &stop);
        if (start + s.size() == stop) {
          kv[k] = s;
          continue;
        }
        kv[k] = s;
        continue;
        errors++;
        LogDetail([l = line_nr](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
          std::stringstream ss;
          ss << "value needs to be integer or double, line=" << l;
          MLPERF_LOG_ERROR(detail, "error_invalid_config", ss.str());
#else
          detail.Error("value needs to be integer or double, line=", l);
#endif
        });
        break;
      }
      if (looking_for == 1 && s != "=") {
        errors++;
        LogDetail([l = line_nr](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
          std::stringstream ss;
          ss << "expected 'key=value', line=" << l;
          MLPERF_LOG_ERROR(detail, "error_invalid_config", ss.str());
#else
          detail.Error("expected 'key=value', line=", l);
#endif
        });
        break;
      }
      if (looking_for == 0) k = s;
      looking_for++;
    }
  }
  if (errors != 0) return -EINVAL;

  uint64_t val;
  std::string val_string;

  // keys that apply to all scenarios
  if (lookupkv(model, scenario, "mode", &val, nullptr)) {
    switch (val) {
      case 0:
        mode = TestMode::SubmissionRun;
        break;
      case 1:
        mode = TestMode::AccuracyOnly;
        break;
      case 2:
        mode = TestMode::PerformanceOnly;
        break;
      case 3:
        mode = TestMode::FindPeakPerformance;
        break;
      default:
        LogDetail([](AsyncDetail &detail) {
#if USE_NEW_LOGGING_FORMAT
          std::stringstream ss;
          ss << "Invalid value passed to Mode key in config.";
          MLPERF_LOG_ERROR(detail, "error_invalid_config", ss.str());
#else
          detail.Error("Invalid value passed to Mode key in config.");
#endif
        });
        break;
    }
  }

  if (conf_type == 0) {
    lookupkv(model, scenario, "qsl_rng_seed", &qsl_rng_seed, nullptr);
    lookupkv(model, scenario, "sample_index_rng_seed", &sample_index_rng_seed,
             nullptr);
    lookupkv(model, scenario, "schedule_rng_seed", &schedule_rng_seed, nullptr);
    lookupkv(model, scenario, "accuracy_log_probability", nullptr,
             &accuracy_log_probability, 0.01);
    if (lookupkv(model, scenario, "test05", &val, nullptr))
      test05 = (val == 1) ? true : false;
    lookupkv(model, scenario, "test05_qsl_rng_seed", &test05_qsl_rng_seed,
             nullptr);
    lookupkv(model, scenario, "test05_sample_index_rng_seed",
             &test05_sample_index_rng_seed, nullptr);
    lookupkv(model, scenario, "test05_schedule_rng_seed",
             &test05_schedule_rng_seed, nullptr);
  }

  // keys that can be overriden in user.conf but will make the results eligible
  // only for open submissions

  // keys to measure token metrics
  if (lookupkv(model, scenario, "use_token_latencies", &val, nullptr)) {
    use_token_latencies = (val == 1) ? true : false;
  }
  if (use_token_latencies) {
    lookupkv(model, "ConstantStream", "ttft_latency", &server_ttft_latency,
             nullptr, 1000 * 1000);
    lookupkv(model, "ConstantStream", "tpot_latency", &server_tpot_latency,
             nullptr, 1000 * 1000);
  }

  // keys to infer token metrics
  if (lookupkv(model, scenario, "infer_token_latencies", &val, nullptr)) {
    infer_token_latencies = (val == 1) ? true : false;
  }
  if (infer_token_latencies) {
    lookupkv(model, scenario, "token_latency_scaling_factor",
             &token_latency_scaling_factor, nullptr, 1);
  }
  // use_grouped_qsl
  if (lookupkv(model, scenario, "use_grouped_qsl", &val, nullptr)) {
    use_grouped_qsl = (val == 1) ? true : false;
  }
  // keys that apply to SingleStream
  lookupkv(model, "SingleStream", "target_latency_percentile", nullptr,
           &single_stream_target_latency_percentile, 0.01);

  // keys that apply to MultiStream
  lookupkv(model, "MultiStream", "target_latency_percentile", nullptr,
           &multi_stream_target_latency_percentile, 0.01);
  lookupkv(model, "MultiStream", "samples_per_query",
           &multi_stream_samples_per_query, nullptr, 1);

  // keys that apply to ConstantStream
  lookupkv(model, "ConstantStream", "target_latency_percentile", nullptr,
           &server_target_latency_percentile, 0.01);

  // keys that can be overriden in user.conf (the provided values still need to
  // pass the submission checker rules)
  if (lookupkv(model, scenario, "performance_issue_unique", &val, nullptr))
    performance_issue_unique = (val == 0) ? false : true;
  if (lookupkv(model, scenario, "performance_issue_same", &val, nullptr))
    performance_issue_same = (val == 0) ? false : true;
  lookupkv(model, scenario, "performance_issue_same_index",
           &performance_issue_same_index, nullptr);

  if (lookupkv(model, scenario, "sample_concatenate_permutation", &val,
               nullptr))
    sample_concatenate_permutation = (val == 1) ? true : false;
  if (lookupkv(model, "ConstantStream", "coalesce_queries", &val, nullptr))
    server_coalesce_queries = (val == 0) ? false : true;
  if (lookupkv(model, "ConstantStream", "max_async_queries", &val, nullptr))
    server_max_async_queries = int(val);
  if (lookupkv(model, "ConstantStream", "constant_gen", &val, nullptr))
    server_constant_gen = (val == 0) ? false : true;

  lookupkv(model, scenario, "min_duration", &min_duration_ms, nullptr);
  lookupkv(model, scenario, "max_duration", &max_duration_ms, nullptr);
  lookupkv(model, scenario, "min_query_count", &min_query_count, nullptr);
  lookupkv(model, scenario, "max_query_count", &max_query_count, nullptr);
  lookupkv(model, scenario, "performance_sample_count_override",
           &performance_sample_count_override, nullptr);
  lookupkv(model, "SingleStream", "target_latency", nullptr,
           &single_stream_expected_latency_ns, 1000 * 1000);
  lookupkv(model, "MultiStream", "target_latency", nullptr,
           &multi_stream_expected_latency_ns, 1000 * 1000);
  lookupkv(model, "ConstantStream", "target_qps", nullptr, &server_target_qps);
  lookupkv(model, "Offline", "target_qps", 0, &offline_expected_qps);

  if (lookupkv(model, scenario, "print_timestamps", &val, nullptr))
    print_timestamps = (val == 0) ? false : true;

  // keys that are used in audit.conf
  lookupkv(model, scenario, "accuracy_log_rng_seed", &accuracy_log_rng_seed,
           nullptr);
  lookupkv(model, scenario, "accuracy_log_sampling_target",
           &accuracy_log_sampling_target, nullptr);
  if (lookupkvstr(model, scenario, "group_sizes", &val_string)) {
    group_sizes.clear();
    size_t pos = 0;
    std::string delimiter = ",";
    std::string token;
    while ((pos = val_string.find(delimiter)) != std::string::npos) {
      token = val_string.substr(0, pos);
      group_sizes.push_back(strtoul(token.c_str(), nullptr, 0));
      val_string.erase(0, pos + delimiter.length());
    }
  }

  return 0;
}

}  // namespace mlperf
