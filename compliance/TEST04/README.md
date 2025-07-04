# Test 04 : Verify SUT is not caching samples
## Introduction

The purpose of this test is to ensure that results are not cached on the fly when SUT sees duplicate sample IDs.

This test requires measuring & comparing performance of SUT in standard (PerformanceOnly, mode=2) mode versus
the following mode:

- TEST04 - Issue same sample: 
	- Offline scenario, the same sample is repeated as many times as necessary to fill the query, targeting a minimum runtime duration of
                                              at least 10 minutes. This breaks the requirement
                                              of reading contiguous memory locations in Offline mode, but it is normal for an audit test, meant to 
                                              stress the SUT in newer ways, to cause performance degradation.
	- Single-Stream/MultiStream/Server scenario test ends after sending the same query as many times as necessary to satisfy a minimum duration of 10 minutes.

## Prerequisites
Test script works best with Python 3.3 or later.

## Exempt Benchmarks
This test is not applicable for the following benchmarks whose performance is dependent on variably sized input samples:
 1. ssd

## Scenarios

 - This test is applicable for all scenarios.

## Pass Criteria
Performance of TEST04 should not be faster than the standard performance run in a statistically significant way. To account for noise, TEST04 can be at most 10% faster than the standard performance run.

## Instructions

### Part II : Run TEST04
 - Copy provided [audit.config](https://github.com/mlperf/mlperf_automotive/blob/master/v0.5/compliance/Organzation/TEST04/audit.config) file in TEST04 folder to the corresponding benchmark directory from where the test is run
 - Run the benchmark
 - Verification that audit.config was properly read can be done by checking that loadgen has found audit.config in `mlperf_log_detail.txt`
 - `mlperf_log_detail.txt` and `mlperf_log_summary.txt` files from this run are required to be submitted under TEST04

### Part III : Compare performance of TEST04 with the standard performance run

 
Check the performance reported by TEST04 with the performance run by running the script below and submit the stdout as `verify_performance.txt` 

	python verify_performance.py -r <mlperf_log_summary.txt generated by performance run> -t <mlperf_log_summary.txt generated by TEST04> | tee verify_performance.txt

Expected outcome:
	`TEST PASS`

Alternatively, the below script can be run which runs the above verification script as well as copies the `mlperf_log_detail.txt` and `mlperf_log_summary.txt` files from the TEST04 run and `verify_performance.txt` output to the output compliance directory for submission:

`python3 run_verification.py -r RESULTS_DIR -c TEST04_DIR -o OUTPUT_DIR`

 - RESULTS_DIR: Specifies path to the directory containing logs from performance run
 - TEST04_DIR: Specifies path to the directory containing logs from compliance test run with TEST04 audit.config
 - OUTPUT_DIR: Specifies the path to the output directory where compliance logs will be uploaded from, i.e. `automotive_results_v0.5/closed/Organization/compliance/bevformer/SingleStream`


Expected outcome:

    Performance check pass: True             
    TEST04 verification complete        


