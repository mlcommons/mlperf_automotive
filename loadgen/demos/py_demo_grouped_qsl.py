# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Python demo showing how to use the MLPerf Inference load generator bindings.
"""

from __future__ import print_function

import threading
import time

import numpy as np
from absl import app
import mlperf_loadgen


def load_samples_to_ram(query_samples):
    del query_samples
    return


def unload_samples_from_ram(query_samples):
    del query_samples
    return


def process_query_async(query_samples):
    time.sleep(0.008)
    responses = []
    for s in query_samples:
        print(s.index)
        responses.append(mlperf_loadgen.QuerySampleResponse(s.id, 0, 0))
    mlperf_loadgen.QuerySamplesComplete(responses)


def issue_query(query_samples):
    threading.Thread(target=process_query_async, args=[query_samples]).start()


def flush_queries():
    pass


def main(argv):
    del argv
    settings = mlperf_loadgen.TestSettings()
    # settings.FromConfig("user.conf", "bevformer", "ConstantStream")
    settings.scenario = mlperf_loadgen.TestScenario.ConstantStream
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.server_target_qps = 100
    settings.server_target_latency_ns = 100000000
    settings.min_query_count = 100
    settings.min_duration_ms = 10000
    settings.server_constant_gen = True
    settings.use_grouped_qsl = True

    sut = mlperf_loadgen.ConstructSUT(issue_query, flush_queries)
    qsl = mlperf_loadgen.ConstructGroupedQSL(
        1024, 32, load_samples_to_ram, unload_samples_from_ram
    )

    # qsl = mlperf_loadgen.ConstructQSL(
    #     1024, 128, load_samples_to_ram, unload_samples_from_ram
    # )
    mlperf_loadgen.StartTestWithGroupedQSL(sut, qsl, settings, "")
    mlperf_loadgen.DestroyGroupedQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)


if __name__ == "__main__":
    app.run(main)
