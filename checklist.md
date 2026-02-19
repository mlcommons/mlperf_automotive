
#### **1. Applicable Categories**
- Insert applicable categories(Eg; adas)

---

#### **2. Applicable Scenarios for Each Category**
- Insert applicable scenarios(Eg; SingleStream,ConstantStream)

---

#### **3. Applicable Compliance Tests**
- Insert Applicable compliance tests

---

#### **5. Validation Dataset: Unique Samples**
Number of **unique samples** in the validation dataset and the QSL size specified in 
- [ ] [inference policies benchmark section](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#41-benchmarks) - (Have to find a place for Automotive)
- [ ] [Automotive benchmark docs](https://github.com/mlcommons/mlperf_automotive/blob/master/docs/index.md)
  *(Ensure QSL size overflows the system cache if possible.)*

---

#### **6. Equal Issue Mode Applicability**
Documented whether **Equal Issue Mode** is applicable in 
- [ ] [mlperf.conf](https://github.com/mlcommons/mlperf_automotive/blob/master/loadgen/mlperf.conf)
- [ ] [Automotive benchmark docs](https://github.com/mlcommons/mlperf_automotive/blob/master/docs/index.md)
  *(Relevant if sample processing times are inconsistent across inputs.)*

---

#### **7. Expected Accuracy and `accuracy.txt` Contents**
- [ ] Detailed expected accuracy and the required contents of the `accuracy.txt` file [here](https://github.com/mlcommons/mlperf_inference_unofficial_submissions_v5.0/blob/auto-update/open/MLCommons/results/mlc-server-reference-gpu-pytorch_v2.2.2-cu124/pointpainting/singlestream/accuracy/accuracy.txt).

---

#### **8. Reference Model Details**
- [ ] Reference model details updated in [Automotive benchmark docs](https://github.com/mlcommons/mlperf_automotive/blob/master/docs/index.md)  

---

#### **9. Reference Implementation Test Coverage**
- [ ] Reference implementation successfully does:
  - [ ] Performance runs
  - [ ] Accuracy runs
  - [ ] Compliance runs (If supported) 


---

#### **10. Test Runs with Smaller Input Sets**
- [ ] Verified the reference implementation can perform test runs with a smaller subset of inputs for:
  - [ ] Performance runs
  - [ ] Accuracy runs

---

#### **11. Dataset and Reference Model Instructions**
- [ ] Hosted the Dataset and Reference model in MLCOMMONS infrastructure
- [ ] Clear instructions provided for:
  - [ ] Downloading the dataset and reference model.
  - [ ] Using the dataset and model for the benchmark.

---

#### **12. Documentation of Recommended System Requirements to run the reference implementation**
- [ ] Added. Eg; [here](https://github.com/mlcommons/inference/blob/docs/docs/system_requirements.yml)

---

#### **13. Submission Checker Modifications**
- [ ] All necessary changes made to the **submission checker** to validate the benchmark.
    - [ ] Create model config for upcoming round
        - [ ] Update name of the benchmark
        - [ ] Update required and optional scenarios for every possible category
        - [ ] Update accuracy target for the benchmark
        - [ ] Update accuracy upper limit(Mostly for LLM's)
        - [ ] Update Performance sample count(The count should be such that the memory needed to load that much dataset should ideally be above a few MBs (> L3 size) but still run on edge systems (not above say 256MB or so))
        - [ ] Update dataset size
        - [ ] Update model mapping (if needed)
        - [ ] Updated seeds for currrent round
        - [ ] Update latency constraints (if any)
        - [ ] Update min queries
        - [ ] Update OFFLINE_MIN_SPQ_SINCE_V4
        - [ ] Update RESULT_FIELD_NEW
        - [ ] Update RESULT_FIELD_BENCHMARK_OVERWRITE(Mostly for LLM's)
        - [ ] Update LLM_LATENCY_LIMITS if any LLM's are introducted
        - [ ] Update accuracy pattern(to capture from accuracy.txt)

---

#### **14. Accuracy Log Truncation Modifications**
- [ ] All necessary changes made to the **truncate_log_accuracy** to validate the benchmark.
    - [ ] Updated values in division and compliance lists.
    - [ ] Fixed conditional statements that are specific to inference for better portability with automotive

---

#### **15. Preprocess submission script Modifications**
- [ ] All necessary changes made to the **preprocess_submission**.
    - [ ] Updated values in divisions list.
    - [ ] Fixed conditional statements that are specific to inference for better portability with automotive
    
---



#### **14. Sample Log Files**
- [ ] Include sample logs for all the applicable scenario runs:
  - [ ] `mlperf_log_summary.txt`
  - [ ] `mlperf_log_detail.txt`  
- [ ] Log files passing the submission checker are generated for all Divisions.
  - [ ] Closed
  - [ ] Open  