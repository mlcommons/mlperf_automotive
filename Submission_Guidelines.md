## Submission Rules

The MLPerf inference submission rules are spread between the [MLCommons policies](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc) and the [MLCommons Inference policies](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) documents. Further, the rules related to power submissions are given [here](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc). The below points are a summary taken from the official rules to act as a checklist for the submitter - please see the original rules for any clarification.


## Hardware requirements
1. MLCommons inference results can be submitted on any hardware and we have past results from Raspberry Pi to high-end inference servers.
2. Closed category submission for adas category needs **ECC RAM** and also needs to have the **networking** capabilities as detailed [here](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#networking-from-the-v30-round)
3. Power submissions need an [approved power analyzer](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc#74-which-power-analyzers-aka-meters-are-supported).

## Things to Know
 
1. Closed submission needs performance and accuracy run for all the required scenarios (as per adas category) with accuracy within 99% or 99.9% as given in the respective task READMEs. Further, the model weights are not supposed to be altered except for quantization. If any of these constraints are not met, the submission cannot go under closed division but can still be submitted under open division.
2. Reference models are mostly fp32 and reference implementations are just for reference and not meant to be directly used by submitters as they are not optimized for performance.
3. Calibration document due **one week** before the submission deadline
4. Power submission needs a power analyzer (approved by SPEC Power) and EULA signature to get access to SPEC PTDaemon
5. To submit under the `available` category your submission system must be available (in whole or in parts and either publicly or to customers) and the software used must be either open source or an **official or beta release** as on the submission deadline. Submissions using nightly release for example cannot be submitted under the available category. 

### Is there an automatic way to run the MLPerf inference benchmarks?

MLPerf inference submissions are expected to be run on various hardware and supported software stacks. Therefore, MLCommons provides only reference implementations to guide submitters in creating optimal implementations for their specific software and hardware configurations. Additionally, all implementations used for MLPerf inference submissions are available in the MLCommons [Inference results](https://github.com/orgs/mlcommons/repositories?q=inference_results_v+sort%3Aname) repositories (under `closed/<submitter>/code` directory), offering further guidance for submitters developing their own implementations.

### Expected time to do benchmark runs
1. Closed submission under adas needs a single stream or constant stream scenario with a minimum of 6636 or 100000 samples, respectively. 
2. Further two compliance runs are needed for closed division taking similar time as the normal performance run.
3. Open division has no accuracy constraints, no required compliance runs, and can be submitted for any single scenario. There is no constraint on the model used except that the model accuracy must be validated on the accuracy dataset used in the corresponding MLPerf inference task [or must be preapproved](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#412-relaxed-constraints-for-the-open-division).


## Validity of the submission

1. [MLCommons automotive submission checker](https://github.com/mlcommons/mlperf_automotive/blob/master/tools/submission/submission_checker.py) is provided to ensure that all submissions are passing the required checks.
2. In the unlikely event that there is an error on the submission checker for your submission, please raise a GitHub issue [here](https://github.com/mlcommons/mlperf_automotive/issues)
3. Any submission passing the submission checker is valid to go to the review discussions but submitters are still required to answer any queries and fix any issues being reported by other submitters.

### Reviewing other submissions
1. Ensure that the `system_desc_id.json` file is having meaningful responses - `submission_checker` only checks for the existence of the fields.
2. README files in the submission directory must be checked to make sure that the instructions are reproducible.
3. Submission checker might be reporting warnings and some of these warnings can warrant an answer from the submitter.



