# MLPerf Automotive Submission Generation Example

This branch contains example results demonstrating submission generation using MLCFlow Automation.

Please note the following points:

- This branch is intended primarily to showcase the recommended submission folder structure that submitters can follow.

- It should not be used as a reference for comparing metrics or analyzing sensitive content that could affect the submission, such as the accuracy log file, as some data has been truncated to reduce size.

- Currently, the branch includes results from the Open Division only.

## Prerequisites

### Clone the Repo
```bash
git clone -b submission-generation-examples https://github.com/mlcommons/mlperf_automotive.git submission_generation_examples --depth 1
```

### Install mlc-scripts
```bash
pip install mlc-scripts
```

### Basic Command(submission_round_5.0)
```bash
mlc run script --tags=generate,mlperf,inference,submission,_wg-automotive \
--results_dir=submission_generation_examples/r0.5 \
--run_checker=yes  \
--submission_dir=my_0.5_submissions  \
--quiet \
--submitter=MLCommons \
--division=open \
--version=v0.5 \
--clean
```

Example Output:
<details>
  
```
  [2025-06-16 16:16:09,199 module.py:575 INFO] - * mlcr generate,inference,submission,_wg-automotive
  [2025-06-16 16:16:09,207 module.py:575 INFO] -   * mlcr get,python3
  [2025-06-16 16:16:09,272 module.py:3886 INFO] -       * /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:09,273 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-python3_ef885f2b
  [2025-06-16 16:16:09,273 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-python3/run.sh from tmp-run.sh
  [2025-06-16 16:16:09,279 module.py:5270 INFO] -              ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-python3/customize.py
  [2025-06-16 16:16:09,281 customize.py:69 INFO] -         Detected version: 3.12.10
  [2025-06-16 16:16:09,282 module.py:3886 INFO] -       * /usr/bin/python3
  [2025-06-16 16:16:09,282 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-python3_ef885f2b
  [2025-06-16 16:16:09,282 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-python3/run.sh from tmp-run.sh
  [2025-06-16 16:16:09,287 module.py:5270 INFO] -              ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-python3/customize.py
  [2025-06-16 16:16:09,289 customize.py:69 INFO] -         Detected version: 3.12.3
  [2025-06-16 16:16:09,289 module.py:3955 INFO] -     Selected 0: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:09,290 module.py:4202 INFO] -       # Found artifact in /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:09,290 module.py:5124 INFO] -          ! cd /home/runner/MLC/repos/local/cache/get-python3_ef885f2b
  [2025-06-16 16:16:09,290 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-python3/run.sh from tmp-run.sh
  [2025-06-16 16:16:09,296 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-python3/customize.py
  [2025-06-16 16:16:09,298 customize.py:69 INFO] -         Detected version: 3.12.10
  [2025-06-16 16:16:09,308 module.py:2165 INFO] -     - cache UID: ef885f2bd40d4c43
  [2025-06-16 16:16:09,308 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:09,308 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:09,315 module.py:575 INFO] -   * mlcr get,sut,system-description
  [2025-06-16 16:16:09,322 module.py:575 INFO] -     * mlcr detect,os
  [2025-06-16 16:16:09,327 module.py:5124 INFO] -            ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:09,327 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:09,347 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:09,369 module.py:575 INFO] -     * mlcr detect,cpu
  [2025-06-16 16:16:09,378 module.py:575 INFO] -       * mlcr detect,os
  [2025-06-16 16:16:09,381 module.py:5124 INFO] -              ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:09,381 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:09,402 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:09,410 module.py:5124 INFO] -            ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:09,410 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/run.sh from tmp-run.sh
  [2025-06-16 16:16:09,459 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/customize.py
  [2025-06-16 16:16:09,469 module.py:575 INFO] -     * mlcr get,python3
  [2025-06-16 16:16:09,470 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:09,470 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:09,471 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:09,476 module.py:575 INFO] -     * mlcr get,compiler,gcc
  [2025-06-16 16:16:09,498 module.py:575 INFO] -       * mlcr detect,os
  [2025-06-16 16:16:09,503 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-gcc_1824fa39
  [2025-06-16 16:16:09,504 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:09,526 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:09,582 module.py:3886 INFO] -         * /usr/bin/gcc
  [2025-06-16 16:16:09,583 module.py:5124 INFO] -                ! cd /home/runner/MLC/repos/local/cache/get-gcc_1824fa39
  [2025-06-16 16:16:09,583 module.py:5125 INFO] -                ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-gcc/run.sh from tmp-run.sh
  /usr/bin/gcc --version
  gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
  Copyright (C) 2023 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  [2025-06-16 16:16:09,594 module.py:5270 INFO] -                ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-gcc/customize.py
  [2025-06-16 16:16:09,618 customize.py:64 INFO] -         Detected version: 13.3.0
  [2025-06-16 16:16:09,618 module.py:3886 INFO] -         * /usr/bin/gcc-12
  [2025-06-16 16:16:09,619 module.py:5124 INFO] -                ! cd /home/runner/MLC/repos/local/cache/get-gcc_1824fa39
  [2025-06-16 16:16:09,619 module.py:5125 INFO] -                ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-gcc/run.sh from tmp-run.sh
  /usr/bin/gcc-12 --version
  gcc-12 (Ubuntu 12.3.0-17ubuntu1) 12.3.0
  Copyright (C) 2022 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  [2025-06-16 16:16:09,945 module.py:5270 INFO] -                ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-gcc/customize.py
  [2025-06-16 16:16:09,969 customize.py:64 INFO] -         Detected version: 12.3.0
  [2025-06-16 16:16:09,969 module.py:3886 INFO] -         * /usr/bin/gcc-14
  [2025-06-16 16:16:09,970 module.py:5124 INFO] -                ! cd /home/runner/MLC/repos/local/cache/get-gcc_1824fa39
  [2025-06-16 16:16:09,970 module.py:5125 INFO] -                ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-gcc/run.sh from tmp-run.sh
  /usr/bin/gcc-14 --version
  gcc-14 (Ubuntu 14.2.0-4ubuntu2~24.04) 14.2.0
  Copyright (C) 2024 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  [2025-06-16 16:16:10,213 module.py:5270 INFO] -                ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-gcc/customize.py
  [2025-06-16 16:16:10,237 customize.py:64 INFO] -         Detected version: 14.2.0
  [2025-06-16 16:16:10,237 module.py:3955 INFO] -       Selected 0: /usr/bin/gcc
  [2025-06-16 16:16:10,238 module.py:4202 INFO] -         # Found artifact in /usr/bin/gcc
  [2025-06-16 16:16:10,238 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-gcc_1824fa39
  [2025-06-16 16:16:10,238 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-gcc/run.sh from tmp-run.sh
  /usr/bin/gcc --version
  gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
  Copyright (C) 2023 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  [2025-06-16 16:16:10,245 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-gcc/customize.py
  [2025-06-16 16:16:10,247 customize.py:64 INFO] -         Detected version: 13.3.0
  [2025-06-16 16:16:10,255 module.py:575 INFO] -     * mlcr get,compiler-flags
  [2025-06-16 16:16:10,261 module.py:575 INFO] -       * mlcr detect,cpu
  [2025-06-16 16:16:10,267 module.py:575 INFO] -         * mlcr detect,os
  [2025-06-16 16:16:10,270 module.py:5124 INFO] -                ! cd /home/runner/MLC/repos/local/cache/get-gcc_1824fa39
  [2025-06-16 16:16:10,270 module.py:5125 INFO] -                ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:10,291 module.py:5270 INFO] -                ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:10,301 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-gcc_1824fa39
  [2025-06-16 16:16:10,301 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/run.sh from tmp-run.sh
  [2025-06-16 16:16:10,342 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/customize.py
  [2025-06-16 16:16:10,606 module.py:2165 INFO] -       - cache UID: 1824fa39b4204fcc
  [2025-06-16 16:16:10,611 module.py:575 INFO] -     * mlcr detect,sudo
  [2025-06-16 16:16:10,624 module.py:5124 INFO] -            ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:10,624 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-sudo/run.sh from tmp-run.sh
  [2025-06-16 16:16:10,628 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-sudo/customize.py
  [2025-06-16 16:16:10,687 module.py:575 INFO] -     * mlcr get,sys-util,generic,_dmidecode
  [2025-06-16 16:16:10,711 module.py:575 INFO] -       * mlcr detect,os
  [2025-06-16 16:16:10,715 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-sys-util_9f1ed0c9
  [2025-06-16 16:16:10,715 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:10,736 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:10,746 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-sys-util_9f1ed0c9
  [2025-06-16 16:16:10,746 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-sys-util/detect.sh from tmp-run.sh
  dmidecode --version > tmp-ver.out
  [2025-06-16 16:16:10,786 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-sys-util/customize.py
  [2025-06-16 16:16:10,790 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-sys-util_9f1ed0c9
  [2025-06-16 16:16:10,790 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-sys-util/detect.sh from tmp-run.sh
  dmidecode --version > tmp-ver.out
  [2025-06-16 16:16:10,796 customize.py:148 INFO] -         Detected version: 3.5
  [2025-06-16 16:16:10,806 module.py:2165 INFO] -       - cache UID: 9f1ed0c9e182498a
  [2025-06-16 16:16:10,812 module.py:575 INFO] -     * mlcr get,cache,dir,_name.mlperf-inference-sut-descriptions
  [2025-06-16 16:16:10,830 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-cache-dir/customize.py
  [2025-06-16 16:16:10,842 module.py:2165 INFO] -       - cache UID: 6b3ccac310b94f0e
  [2025-06-16 16:16:10,845 customize.py:56 INFO] - Generating SUT description file for pkrvmxyh4eaekms
  HW description file for pkrvmxyh4eaekms not found. Copying from default!!!
  [2025-06-16 16:16:10,846 module.py:5124 INFO] -          ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:10,846 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-mlperf-inference-sut-description/detect_memory.sh from tmp-run.sh
  [2025-06-16 16:16:10,861 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-mlperf-inference-sut-description/customize.py
  [2025-06-16 16:16:10,866 module.py:575 INFO] -   * mlcr parse,dmidecode,memory,info
  [2025-06-16 16:16:10,938 module.py:575 INFO] -     * mlcr get,generic-python-lib,_package.dmiparser
  [2025-06-16 16:16:10,962 module.py:575 INFO] -       * mlcr detect,os
  [2025-06-16 16:16:10,965 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_636118e1
  [2025-06-16 16:16:10,965 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:10,987 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:10,998 module.py:575 INFO] -       * mlcr detect,cpu
  [2025-06-16 16:16:11,004 module.py:575 INFO] -         * mlcr detect,os
  [2025-06-16 16:16:11,008 module.py:5124 INFO] -                ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_636118e1
  [2025-06-16 16:16:11,008 module.py:5125 INFO] -                ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:11,032 module.py:5270 INFO] -                ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:11,041 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_636118e1
  [2025-06-16 16:16:11,041 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/run.sh from tmp-run.sh
  [2025-06-16 16:16:11,082 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/customize.py
  [2025-06-16 16:16:11,093 module.py:575 INFO] -       * mlcr get,python3
  [2025-06-16 16:16:11,093 module.py:1284 INFO] -            ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:11,094 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:11,094 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:11,141 module.py:575 INFO] -       * mlcr get,generic-python-lib,_pip
  [2025-06-16 16:16:11,161 module.py:575 INFO] -         * mlcr detect,os
  [2025-06-16 16:16:11,165 module.py:5124 INFO] -                ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_5c7d1922
  [2025-06-16 16:16:11,165 module.py:5125 INFO] -                ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:11,186 module.py:5270 INFO] -                ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:11,197 module.py:575 INFO] -         * mlcr detect,cpu
  [2025-06-16 16:16:11,202 module.py:575 INFO] -           * mlcr detect,os
  [2025-06-16 16:16:11,206 module.py:5124 INFO] -                  ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_5c7d1922
  [2025-06-16 16:16:11,206 module.py:5125 INFO] -                  ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:11,227 module.py:5270 INFO] -                  ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:11,236 module.py:5124 INFO] -                ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_5c7d1922
  [2025-06-16 16:16:11,237 module.py:5125 INFO] -                ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/run.sh from tmp-run.sh
  [2025-06-16 16:16:11,278 module.py:5270 INFO] -                ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/customize.py
  [2025-06-16 16:16:11,288 module.py:575 INFO] -         * mlcr get,python3
  [2025-06-16 16:16:11,289 module.py:1284 INFO] -              ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:11,289 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:11,290 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:11,293 module.py:5124 INFO] -                  ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_5c7d1922
  [2025-06-16 16:16:11,294 module.py:5125 INFO] -                  ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:11,368 module.py:5270 INFO] -                  ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:11,372 customize.py:152 INFO] -             Detected version: 25.1.1
  [2025-06-16 16:16:11,372 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_5c7d1922
  [2025-06-16 16:16:11,372 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:11,446 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:11,468 module.py:2165 INFO] -         - cache UID: 5c7d1922d65b49f2
  [2025-06-16 16:16:11,476 module.py:5124 INFO] -                ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_636118e1
  [2025-06-16 16:16:11,476 module.py:5125 INFO] -                ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:11,549 module.py:5270 INFO] -                ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:11,556 customize.py:117 INFO] -           Extra PIP CMD:   --break-system-packages 
  [2025-06-16 16:16:11,557 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_636118e1
  [2025-06-16 16:16:11,557 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/install.sh from tmp-run.sh
  
  /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3 -m pip install "dmiparser" --break-system-packages
  Collecting dmiparser
    Downloading dmiparser-5.1-py3-none-any.whl.metadata (5.7 kB)
  Downloading dmiparser-5.1-py3-none-any.whl (8.3 kB)
  Installing collected packages: dmiparser
  Successfully installed dmiparser-5.1
  [2025-06-16 16:16:12,644 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_636118e1
  [2025-06-16 16:16:12,644 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:12,717 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:12,722 customize.py:152 INFO] -           Detected version: 5.1
  [2025-06-16 16:16:12,733 module.py:2165 INFO] -       - cache UID: 636118e12e664276
  [2025-06-16 16:16:12,737 module.py:5124 INFO] -          ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:12,737 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/parse-dmidecode-memory-info/run.sh from tmp-run.sh
  Running: 
  /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3 /home/runner/MLC/repos/mlcommons@mlperf-automations/script/parse-dmidecode-memory-info/get_memory_info.py /home/runner/work/mlperf_automotive/mlperf_automotive/meminfo.dump /home/runner/work/mlperf_automotive/mlperf_automotive/meminfo.txt
  
  [2025-06-16 16:16:12,808 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/parse-dmidecode-memory-info/customize.py
  [2025-06-16 16:16:12,819 module.py:575 INFO] -   * mlcr install,pip-package,for-mlc-python,_package.tabulate
  Requirement already satisfied: tabulate in /opt/hostedtoolcache/Python/3.12.10/x64/lib/python3.12/site-packages (0.9.0)
  [2025-06-16 16:16:13,925 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/install-pip-package-for-mlc-python/customize.py
  [2025-06-16 16:16:13,938 module.py:2165 INFO] -     - cache UID: acb42a4647cc4353
  [2025-06-16 16:16:13,947 module.py:575 INFO] -   * mlcr mlcommons,automotive,src
  [2025-06-16 16:16:13,968 module.py:575 INFO] -     * mlcr detect,os
  [2025-06-16 16:16:13,971 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-mlperf-automotive-src_01435494
  [2025-06-16 16:16:13,971 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:13,991 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:14,004 module.py:575 INFO] -     * mlcr get,python3
  [2025-06-16 16:16:14,005 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:14,005 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:14,005 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:14,017 module.py:575 INFO] -     * mlcr get,git,repo,_branch.master,_repo.https://github.com/mlcommons/mlperf_automotive
  [2025-06-16 16:16:14,040 module.py:575 INFO] -       * mlcr detect,os
  [2025-06-16 16:16:14,044 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09
  [2025-06-16 16:16:14,044 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:14,064 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:14,073 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09
  [2025-06-16 16:16:14,073 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-git-repo/run.sh from tmp-run.sh
  /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09
  rm -rf automotive
  ******************************************************
  Current directory: /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09
  
  Cloning mlperf_automotive from https://github.com/mlcommons/mlperf_automotive
  
  git clone  -b master https://github.com/mlcommons/mlperf_automotive --depth 5 automotive
  
  Cloning into 'automotive'...
  git rev-parse HEAD >> ../tmp-mlc-git-hash.out
  [2025-06-16 16:16:14,770 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-git-repo/customize.py
  [2025-06-16 16:16:14,783 module.py:2165 INFO] -       - cache UID: 4c2f5d0975564011
  [2025-06-16 16:16:14,783 module.py:2240 INFO] - MLC cache path to the Git repo: /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive
  [2025-06-16 16:16:14,784 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-mlperf-automotive-src/customize.py
  [2025-06-16 16:16:14,796 module.py:2165 INFO] -     - cache UID: 014354942ec544dc
  [2025-06-16 16:16:14,796 module.py:2240 INFO] - Path to MLPerf automotive benchmark source: /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive
  [2025-06-16 16:16:14,801 module.py:575 INFO] -   * mlcr get,mlperf,automotive,utils
  [2025-06-16 16:16:14,811 module.py:575 INFO] -     * mlcr get,mlperf,automotive,src
  [2025-06-16 16:16:14,812 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-mlperf-automotive-src_01435494/mlc-cached-state.json
  [2025-06-16 16:16:14,812 module.py:2240 INFO] - Path to MLPerf automotive benchmark source: /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive
  [2025-06-16 16:16:14,815 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-mlperf-automotive-utils/customize.py
  [2025-06-16 16:16:14,846 module.py:5270 INFO] -        ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/generate-mlperf-inference-submission/customize.py
  [2025-06-16 16:16:14,848 customize.py:87 INFO] - =================================================
  Cleaning mysubmissions/mlperf_submission ...
  [2025-06-16 16:16:14,848 customize.py:93 INFO] - =================================================
  [2025-06-16 16:16:14,848 customize.py:101 INFO] - * MLPerf inference submission dir: mysubmissions/mlperf_submission
  [2025-06-16 16:16:14,848 customize.py:102 INFO] - * MLPerf inference results dir: /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/
  [2025-06-16 16:16:14,848 customize.py:140 INFO] - * MLPerf inference division: open
  [2025-06-16 16:16:14,849 customize.py:155 INFO] - * MLPerf inference submitter: MLCommons
  [2025-06-16 16:16:14,849 customize.py:176 INFO] - Created calibration.md file at mysubmissions/mlperf_submission/open/MLCommons/calibration.md
  [2025-06-16 16:16:14,849 customize.py:182 INFO] - Created calibration.md file at mysubmissions/mlperf_submission/open/MLCommons/calibration.md
  sut info completely filled from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/d42777903298-reference-gpu-pytorch-v2.3.1-cu124/mlc-sut-info.json!
  [2025-06-16 16:16:14,849 customize.py:299 INFO] - The SUT folder name for submission generation is: pkrvmxyh4eaekms-reference-gpu-pytorch-cu124
  [2025-06-16 16:16:14,850 customize.py:357 INFO] - * MLPerf inference model: ssd
  [2025-06-16 16:16:14,851 customize.py:656 INFO] -  * mlperf_log_accuracy.json
  [2025-06-16 16:16:14,851 customize.py:656 INFO] -  * mlperf_log_detail.txt
  [2025-06-16 16:16:14,851 customize.py:656 INFO] -  * mlperf_log_summary.txt
  [2025-06-16 16:16:14,851 customize.py:656 INFO] -  * accuracy.txt
  [2025-06-16 16:16:14,852 customize.py:505 INFO] - {'accelerator_frequency': '2040000 MHz', 'accelerator_host_interconnect': 'N/A', 'accelerator_interconnect': 'N/A', 'accelerator_interconnect_topology': '', 'accelerator_memory_capacity': '21.95147705078125 GB', 'accelerator_memory_configuration': 'N/A', 'accelerator_model_name': 'NVIDIA L4', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 1, 'cooling': 'air', 'division': 'open', 'framework': 'pytorch', 'host_memory_capacity': '128G', 'host_memory_configuration': 'undefined', 'host_network_card_count': '1', 'host_networking': 'Gig Ethernet', 'host_networking_topology': 'N/A', 'host_processor_caches': 'L1d cache: 512 KiB (16 instances), L1i cache: 512 KiB (16 instances), L2 cache: 16 MiB (16 instances), L3 cache: 38.5 MiB (1 instance)', 'host_processor_core_count': '16', 'host_processor_frequency': 'undefined', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) CPU @ 2.20GHz', 'host_processors_per_node': '1', 'host_storage_capacity': '6.1T', 'host_storage_type': 'SSD', 'hw_notes': '', 'number_of_nodes': '1', 'operating_system': 'Ubuntu 22.04 (linux-6.8.0-1030-gcp-glibc2.35)', 'other_software_stack': 'Python: 3.10.12, LLVM-15.0.6, Using Docker  , CUDA 12.4', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'd42777903298', 'system_type': 'edge', 'system_type_detail': 'edge server'}
  [2025-06-16 16:16:14,853 customize.py:656 INFO] -  * mlperf_log_accuracy.json
  [2025-06-16 16:16:14,853 customize.py:656 INFO] -  * mlperf_log_detail.txt
  [2025-06-16 16:16:14,854 customize.py:656 INFO] -  * mlperf_log_summary.txt
  [2025-06-16 16:16:14,854 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/d42777903298-reference-gpu-pytorch-v2.3.1-cu124/ssd/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:14,855 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/d42777903298-reference-gpu-pytorch-v2.3.1-cu124/ssd/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:14,855 submission_checker.py:729 INFO] v0.5, SingleStream
  [2025-06-16 16:16:14,856 customize.py:740 INFO] - +-------+--------------+----------+------------+-----------------+
  | Model |   Scenario   | Accuracy | Throughput | Latency (in ms) |
  +-------+--------------+----------+------------+-----------------+
  |  ssd  | singlestream | 0.71801  |    1.72    |     581.485     |
  +-------+--------------+----------+------------+-----------------+
  sut info completely filled from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/2a3adbd04241-reference-cpu-onnxruntime-v1.22.0-default_config/mlc-sut-info.json!
  [2025-06-16 16:16:14,856 customize.py:299 INFO] - The SUT folder name for submission generation is: pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config
  [2025-06-16 16:16:14,857 customize.py:357 INFO] - * MLPerf inference model: bevformer
  [2025-06-16 16:16:14,858 customize.py:656 INFO] -  * mlperf_log_accuracy.json
  [2025-06-16 16:16:14,858 customize.py:656 INFO] -  * mlperf_log_detail.txt
  [2025-06-16 16:16:14,858 customize.py:656 INFO] -  * mlperf_log_summary.txt
  [2025-06-16 16:16:14,859 customize.py:656 INFO] -  * accuracy.txt
  [2025-06-16 16:16:14,859 customize.py:505 INFO] - {'accelerator_frequency': '', 'accelerator_host_interconnect': 'N/A', 'accelerator_interconnect': 'N/A', 'accelerator_interconnect_topology': '', 'accelerator_memory_capacity': 'N/A', 'accelerator_memory_configuration': 'N/A', 'accelerator_model_name': 'N/A', 'accelerator_on-chip_memories': '', 'accelerators_per_node': '0', 'cooling': 'air', 'division': 'open', 'framework': 'onnxruntime', 'host_memory_capacity': '128G', 'host_memory_configuration': 'undefined', 'host_network_card_count': '1', 'host_networking': 'Gig Ethernet', 'host_networking_topology': 'N/A', 'host_processor_caches': 'L1d cache: 512 KiB (16 instances), L1i cache: 512 KiB (16 instances), L2 cache: 16 MiB (16 instances), L3 cache: 38.5 MiB (1 instance)', 'host_processor_core_count': '16', 'host_processor_frequency': 'undefined', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) CPU @ 2.20GHz', 'host_processors_per_node': '1', 'host_storage_capacity': '6.0T', 'host_storage_type': 'SSD', 'hw_notes': '', 'number_of_nodes': '1', 'operating_system': 'Ubuntu 22.04 (linux-6.8.0-1030-gcp-glibc2.35)', 'other_software_stack': 'Python: 3.10.12, LLVM-15.0.6, Using Docker ', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': '2a3adbd04241', 'system_type': 'edge', 'system_type_detail': 'edge server'}
  [2025-06-16 16:16:14,860 customize.py:656 INFO] -  * mlperf_log_detail.txt
  [2025-06-16 16:16:14,861 customize.py:656 INFO] -  * mlperf_log_accuracy-tmo.json
  [2025-06-16 16:16:14,861 customize.py:656 INFO] -  * mlperf_log_summary.txt
  [2025-06-16 16:16:14,862 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/2a3adbd04241-reference-cpu-onnxruntime-v1.22.0-default_config/bevformer/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:14,862 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/2a3adbd04241-reference-cpu-onnxruntime-v1.22.0-default_config/bevformer/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:14,862 submission_checker.py:729 INFO] v0.5, SingleStream
  [2025-06-16 16:16:14,863 customize.py:740 INFO] - +-----------+--------------+----------+------------+-----------------+
  |   Model   |   Scenario   | Accuracy | Throughput | Latency (in ms) |
  +-----------+--------------+----------+------------+-----------------+
  | bevformer | singlestream |  0.2684  |   0.155    |    6444.096     |
  +-----------+--------------+----------+------------+-----------------+
  sut info completely filled from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/792f6f734930-reference-cpu-onnxruntime-v1.22.0-default_config/mlc-sut-info.json!
  [2025-06-16 16:16:14,864 customize.py:299 INFO] - The SUT folder name for submission generation is: pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config
  [2025-06-16 16:16:14,864 customize.py:357 INFO] - * MLPerf inference model: deeplabv3plus
  [2025-06-16 16:16:14,865 customize.py:656 INFO] -  * mlperf_log_accuracy.json
  [2025-06-16 16:16:14,865 customize.py:656 INFO] -  * mlperf_log_detail.txt
  [2025-06-16 16:16:14,866 customize.py:656 INFO] -  * mlperf_log_summary.txt
  [2025-06-16 16:16:14,866 customize.py:656 INFO] -  * accuracy.txt
  [2025-06-16 16:16:14,866 customize.py:505 INFO] - {'accelerator_frequency': '', 'accelerator_host_interconnect': 'N/A', 'accelerator_interconnect': 'N/A', 'accelerator_interconnect_topology': '', 'accelerator_memory_capacity': 'N/A', 'accelerator_memory_configuration': 'N/A', 'accelerator_model_name': 'N/A', 'accelerator_on-chip_memories': '', 'accelerators_per_node': '0', 'cooling': 'air', 'division': 'open', 'framework': 'onnxruntime', 'host_memory_capacity': '128G', 'host_memory_configuration': 'undefined', 'host_network_card_count': '1', 'host_networking': 'Gig Ethernet', 'host_networking_topology': 'N/A', 'host_processor_caches': 'L1d cache: 512 KiB (16 instances), L1i cache: 512 KiB (16 instances), L2 cache: 16 MiB (16 instances), L3 cache: 38.5 MiB (1 instance)', 'host_processor_core_count': '16', 'host_processor_frequency': 'undefined', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) CPU @ 2.20GHz', 'host_processors_per_node': '1', 'host_storage_capacity': '6.0T', 'host_storage_type': 'SSD', 'hw_notes': '', 'number_of_nodes': '1', 'operating_system': 'Ubuntu 22.04 (linux-6.8.0-1030-gcp-glibc2.35)', 'other_software_stack': 'Python: 3.10.12, LLVM-15.0.6, Using Docker ', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': '792f6f734930', 'system_type': 'edge', 'system_type_detail': 'edge server'}
  [2025-06-16 16:16:14,867 customize.py:656 INFO] -  * mlperf_log_accuracy.json
  [2025-06-16 16:16:14,868 customize.py:656 INFO] -  * mlperf_log_detail.txt
  [2025-06-16 16:16:14,868 customize.py:656 INFO] -  * mlperf_log_summary.txt
  [2025-06-16 16:16:14,869 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/792f6f734930-reference-cpu-onnxruntime-v1.22.0-default_config/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:14,869 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/792f6f734930-reference-cpu-onnxruntime-v1.22.0-default_config/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:14,869 submission_checker.py:729 INFO] v0.5, SingleStream
  [2025-06-16 16:16:14,870 customize.py:740 INFO] - +---------------+--------------+----------+------------+-----------------+
  |     Model     |   Scenario   | Accuracy | Throughput | Latency (in ms) |
  +---------------+--------------+----------+------------+-----------------+
  | deeplabv3plus | singlestream | 0.92436  |   0.061    |    16265.09     |
  +---------------+--------------+----------+------------+-----------------+
  sut info completely filled from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/e094ed31695a-reference-gpu-pytorch-v2.3.1-cu124/mlc-sut-info.json!
  [2025-06-16 16:16:14,871 customize.py:299 INFO] - The SUT folder name for submission generation is: pkrvmxyh4eaekms-reference-gpu-pytorch-cu124
  [2025-06-16 16:16:14,871 customize.py:357 INFO] - * MLPerf inference model: deeplabv3plus
  [2025-06-16 16:16:14,872 customize.py:656 INFO] -  * mlperf_log_accuracy.json
  [2025-06-16 16:16:14,872 customize.py:656 INFO] -  * mlperf_log_detail.txt
  [2025-06-16 16:16:14,873 customize.py:656 INFO] -  * mlperf_log_summary.txt
  [2025-06-16 16:16:14,873 customize.py:656 INFO] -  * accuracy.txt
  [2025-06-16 16:16:14,873 customize.py:505 INFO] - {'accelerator_frequency': '2040000 MHz', 'accelerator_host_interconnect': 'N/A', 'accelerator_interconnect': 'N/A', 'accelerator_interconnect_topology': '', 'accelerator_memory_capacity': '21.95147705078125 GB', 'accelerator_memory_configuration': 'N/A', 'accelerator_model_name': 'NVIDIA L4', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 1, 'cooling': 'air', 'division': 'open', 'framework': 'pytorch', 'host_memory_capacity': '128G', 'host_memory_configuration': 'undefined', 'host_network_card_count': '1', 'host_networking': 'Gig Ethernet', 'host_networking_topology': 'N/A', 'host_processor_caches': 'L1d cache: 512 KiB (16 instances), L1i cache: 512 KiB (16 instances), L2 cache: 16 MiB (16 instances), L3 cache: 38.5 MiB (1 instance)', 'host_processor_core_count': '16', 'host_processor_frequency': 'undefined', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) CPU @ 2.20GHz', 'host_processors_per_node': '1', 'host_storage_capacity': '6.1T', 'host_storage_type': 'SSD', 'hw_notes': '', 'number_of_nodes': '1', 'operating_system': 'Ubuntu 22.04 (linux-6.8.0-1030-gcp-glibc2.35)', 'other_software_stack': 'Python: 3.10.12, LLVM-15.0.6, Using Docker  , CUDA 12.4', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'e094ed31695a', 'system_type': 'edge', 'system_type_detail': 'edge server'}
  [2025-06-16 16:16:14,875 customize.py:656 INFO] -  * mlperf_log_accuracy.json
  [2025-06-16 16:16:14,875 customize.py:656 INFO] -  * mlperf_log_detail.txt
  [2025-06-16 16:16:14,875 customize.py:656 INFO] -  * mlperf_log_summary.txt
  [2025-06-16 16:16:14,876 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/e094ed31695a-reference-gpu-pytorch-v2.3.1-cu124/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:14,877 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from /home/runner/work/mlperf_automotive/mlperf_automotive/submission_generation_examples/r0.5/e094ed31695a-reference-gpu-pytorch-v2.3.1-cu124/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:14,877 submission_checker.py:729 INFO] v0.5, SingleStream
  [2025-06-16 16:16:14,878 customize.py:740 INFO] - +---------------+--------------+----------+------------+-----------------+
  |     Model     |   Scenario   | Accuracy | Throughput | Latency (in ms) |
  +---------------+--------------+----------+------------+-----------------+
  | deeplabv3plus | singlestream | 0.92436  |   1.672    |     598.042     |
  +---------------+--------------+----------+------------+-----------------+
  [2025-06-16 16:16:14,885 module.py:575 INFO] - * mlcr accuracy,truncate,mlc,_wg-automotive
  [2025-06-16 16:16:14,892 module.py:575 INFO] -   * mlcr get,python3
  [2025-06-16 16:16:14,893 module.py:1284 INFO] -        ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:14,893 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:14,893 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:14,903 module.py:575 INFO] -   * mlcr mlcommons,automotive,src
  [2025-06-16 16:16:14,904 module.py:1284 INFO] -        ! load /home/runner/MLC/repos/local/cache/get-mlperf-automotive-src_01435494/mlc-cached-state.json
  [2025-06-16 16:16:14,904 module.py:2240 INFO] - Path to MLPerf automotive benchmark source: /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive
  [2025-06-16 16:16:14,909 module.py:5124 INFO] -        ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:14,909 module.py:5125 INFO] -        ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/truncate-mlperf-inference-accuracy-log/run.sh from tmp-run.sh
  python3 '/home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive/tools/submission/truncate_accuracy_log.py' --input 'mysubmissions/mlperf_submission' --submitter 'MLCommons' --backup 'mysubmissions/mlperf_submission_logs'
  INFO:main:open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/deeplabv3plus/singlestream/accuracy/mlperf_log_accuracy.json truncated
  INFO:main:open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/bevformer/singlestream/accuracy/mlperf_log_accuracy.json truncated
  INFO:main:open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/ssd/singlestream/accuracy/mlperf_log_accuracy.json truncated
  INFO:main:open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/deeplabv3plus/singlestream/accuracy/mlperf_log_accuracy.json truncated
  INFO:main:Make sure you keep a backup of mysubmissions/mlperf_submission_logs in case mlperf wants to see the original accuracy logs
  [2025-06-16 16:16:14,964 module.py:575 INFO] - * mlcr preprocess,mlperf,submission,_wg-automotive
  [2025-06-16 16:16:14,973 module.py:575 INFO] -   * mlcr get,python3
  [2025-06-16 16:16:14,973 module.py:1284 INFO] -        ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:14,974 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:14,974 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:14,983 module.py:575 INFO] -   * mlcr mlcommons,automotive,src
  [2025-06-16 16:16:14,984 module.py:1284 INFO] -        ! load /home/runner/MLC/repos/local/cache/get-mlperf-automotive-src_01435494/mlc-cached-state.json
  [2025-06-16 16:16:14,984 module.py:2240 INFO] - Path to MLPerf automotive benchmark source: /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive
  [2025-06-16 16:16:14,987 module.py:5124 INFO] -        ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:14,987 module.py:5125 INFO] -        ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/preprocess-mlperf-inference-submission/run.sh from tmp-run.sh
  python3 '/home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive/tools/submission/preprocess_submission.py' --input 'mysubmissions/mlperf_submission' --submitter 'MLCommons' --output 'mysubmissions/mlperf_submission_processed' --version v0.5  --noinfer-low-accuracy-results --noinfer-scenario-results
  [2025-06-16 16:16:15,046 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/deeplabv3plus/singlestream/accuracy/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,047 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,047 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,050 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/bevformer/singlestream/accuracy/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,051 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/bevformer/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,052 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/bevformer/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,052 submission_checker.py:797 ERROR] open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/bevformer/singlestream/performance/run_1/mlperf_log_detail.txt performance_sample_count, found 128, needs to be >= 512
  [2025-06-16 16:16:15,052 preprocess_submission.py:272 WARNING] singlestream scenario result is invalid for pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config: bevformer in open division. Accuracy: True, Performance: False. Removing...
  [2025-06-16 16:16:15,056 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/ssd/singlestream/accuracy/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,057 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/ssd/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,058 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/ssd/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,061 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/deeplabv3plus/singlestream/accuracy/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,062 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,062 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:15,063 preprocess_submission.py:92 INFO] Removing empty dir: (mysubmissions/mlperf_submission_processed/open/MLCommons/measurements/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/bevformer)
  [2025-06-16 16:16:15,064 preprocess_submission.py:92 INFO] Removing empty dir: (mysubmissions/mlperf_submission_processed/open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/bevformer)
  [2025-06-16 16:16:15,071 module.py:5270 INFO] -        ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/preprocess-mlperf-inference-submission/customize.py
  [2025-06-16 16:16:15,107 module.py:575 INFO] - * mlcr submission,inference,checker,mlc,_wg-automotive
  [2025-06-16 16:16:15,116 module.py:575 INFO] -   * mlcr get,python3
  [2025-06-16 16:16:15,116 module.py:1284 INFO] -        ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:15,117 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:15,117 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:15,164 module.py:575 INFO] -   * mlcr get,generic-python-lib,_xlsxwriter
  [2025-06-16 16:16:15,186 module.py:575 INFO] -     * mlcr detect,os
  [2025-06-16 16:16:15,190 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_95e6139b
  [2025-06-16 16:16:15,190 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:15,211 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:15,223 module.py:575 INFO] -     * mlcr detect,cpu
  [2025-06-16 16:16:15,229 module.py:575 INFO] -       * mlcr detect,os
  [2025-06-16 16:16:15,234 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_95e6139b
  [2025-06-16 16:16:15,234 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:15,255 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:15,266 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_95e6139b
  [2025-06-16 16:16:15,266 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/run.sh from tmp-run.sh
  [2025-06-16 16:16:15,308 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/customize.py
  [2025-06-16 16:16:15,326 module.py:575 INFO] -     * mlcr get,python3
  [2025-06-16 16:16:15,327 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:15,327 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:15,327 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:15,585 module.py:575 INFO] -     * mlcr get,generic-python-lib,_pip
  [2025-06-16 16:16:15,594 module.py:575 INFO] -       * mlcr get,python3
  [2025-06-16 16:16:15,595 module.py:1284 INFO] -            ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:15,595 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:15,596 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:15,596 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_95e6139b
  [2025-06-16 16:16:15,596 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/validate_cache.sh from tmp-run.sh
  [2025-06-16 16:16:15,671 module.py:5270 INFO] -            ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:15,678 customize.py:152 INFO] -           Detected version: 25.1.1
  [2025-06-16 16:16:15,686 module.py:575 INFO] -       * mlcr get,python3
  [2025-06-16 16:16:15,688 module.py:1284 INFO] -            ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:15,689 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:15,689 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:15,690 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-generic-python-lib_5c7d1922/mlc-cached-state.json
  [2025-06-16 16:16:15,694 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_95e6139b
  [2025-06-16 16:16:15,694 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:15,768 module.py:5270 INFO] -              ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:15,772 customize.py:117 INFO] -         Extra PIP CMD:   --break-system-packages 
  [2025-06-16 16:16:15,773 module.py:5124 INFO] -          ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_95e6139b
  [2025-06-16 16:16:15,773 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/install.sh from tmp-run.sh
  
  /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3 -m pip install "xlsxwriter" --break-system-packages
  Collecting xlsxwriter
    Downloading XlsxWriter-3.2.3-py3-none-any.whl.metadata (2.7 kB)
  Downloading XlsxWriter-3.2.3-py3-none-any.whl (169 kB)
  Installing collected packages: xlsxwriter
  Successfully installed xlsxwriter-3.2.3
  [2025-06-16 16:16:16,617 module.py:5124 INFO] -          ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_95e6139b
  [2025-06-16 16:16:16,617 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:16,691 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:16,695 customize.py:152 INFO] -         Detected version: 3.2.3
  [2025-06-16 16:16:16,705 module.py:2165 INFO] -     - cache UID: 95e6139bd7fe43de
  [2025-06-16 16:16:16,752 module.py:575 INFO] -   * mlcr get,generic-python-lib,_package.pyarrow
  [2025-06-16 16:16:16,775 module.py:575 INFO] -     * mlcr detect,os
  [2025-06-16 16:16:16,778 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_6cb05234
  [2025-06-16 16:16:16,778 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:16,799 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:16,811 module.py:575 INFO] -     * mlcr detect,cpu
  [2025-06-16 16:16:16,818 module.py:575 INFO] -       * mlcr detect,os
  [2025-06-16 16:16:16,823 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_6cb05234
  [2025-06-16 16:16:16,823 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:16,844 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:16,855 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_6cb05234
  [2025-06-16 16:16:16,855 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/run.sh from tmp-run.sh
  [2025-06-16 16:16:16,894 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/customize.py
  [2025-06-16 16:16:16,912 module.py:575 INFO] -     * mlcr get,python3
  [2025-06-16 16:16:16,913 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:16,913 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:16,913 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:17,167 module.py:575 INFO] -     * mlcr get,generic-python-lib,_pip
  [2025-06-16 16:16:17,176 module.py:575 INFO] -       * mlcr get,python3
  [2025-06-16 16:16:17,177 module.py:1284 INFO] -            ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:17,177 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:17,178 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:17,178 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_6cb05234
  [2025-06-16 16:16:17,178 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/validate_cache.sh from tmp-run.sh
  [2025-06-16 16:16:17,253 module.py:5270 INFO] -            ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:17,260 customize.py:152 INFO] -           Detected version: 25.1.1
  [2025-06-16 16:16:17,268 module.py:575 INFO] -       * mlcr get,python3
  [2025-06-16 16:16:17,270 module.py:1284 INFO] -            ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:17,271 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:17,271 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:17,271 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-generic-python-lib_5c7d1922/mlc-cached-state.json
  [2025-06-16 16:16:17,276 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_6cb05234
  [2025-06-16 16:16:17,276 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:17,351 module.py:5270 INFO] -              ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:17,355 customize.py:117 INFO] -         Extra PIP CMD:   --break-system-packages 
  [2025-06-16 16:16:17,356 module.py:5124 INFO] -          ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_6cb05234
  [2025-06-16 16:16:17,356 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/install.sh from tmp-run.sh
  
  /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3 -m pip install "pyarrow" --break-system-packages
  Collecting pyarrow
    Downloading pyarrow-20.0.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (3.3 kB)
  Downloading pyarrow-20.0.0-cp312-cp312-manylinux_2_28_x86_64.whl (42.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.3/42.3 MB 183.9 MB/s eta 0:00:00
  Installing collected packages: pyarrow
  Successfully installed pyarrow-20.0.0
  [2025-06-16 16:16:19,490 module.py:5124 INFO] -          ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_6cb05234
  [2025-06-16 16:16:19,490 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:19,563 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:19,567 customize.py:152 INFO] -         Detected version: 20.0.0
  [2025-06-16 16:16:19,577 module.py:2165 INFO] -     - cache UID: 6cb05234962c4bc0
  [2025-06-16 16:16:19,623 module.py:575 INFO] -   * mlcr get,generic-python-lib,_pandas
  [2025-06-16 16:16:19,646 module.py:575 INFO] -     * mlcr detect,os
  [2025-06-16 16:16:19,649 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_0bd44e38
  [2025-06-16 16:16:19,650 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:19,670 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:19,682 module.py:575 INFO] -     * mlcr detect,cpu
  [2025-06-16 16:16:19,688 module.py:575 INFO] -       * mlcr detect,os
  [2025-06-16 16:16:19,693 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_0bd44e38
  [2025-06-16 16:16:19,693 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
  [2025-06-16 16:16:19,714 module.py:5270 INFO] -              ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-os/customize.py
  [2025-06-16 16:16:19,725 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_0bd44e38
  [2025-06-16 16:16:19,725 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/run.sh from tmp-run.sh
  [2025-06-16 16:16:19,765 module.py:5270 INFO] -            ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/detect-cpu/customize.py
  [2025-06-16 16:16:19,783 module.py:575 INFO] -     * mlcr get,python3
  [2025-06-16 16:16:19,784 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:19,784 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:19,784 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:20,038 module.py:575 INFO] -     * mlcr get,generic-python-lib,_pip
  [2025-06-16 16:16:20,047 module.py:575 INFO] -       * mlcr get,python3
  [2025-06-16 16:16:20,048 module.py:1284 INFO] -            ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:20,048 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:20,048 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:20,049 module.py:5124 INFO] -            ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_0bd44e38
  [2025-06-16 16:16:20,049 module.py:5125 INFO] -            ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/validate_cache.sh from tmp-run.sh
  [2025-06-16 16:16:20,124 module.py:5270 INFO] -            ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:20,132 customize.py:152 INFO] -           Detected version: 25.1.1
  [2025-06-16 16:16:20,140 module.py:575 INFO] -       * mlcr get,python3
  [2025-06-16 16:16:20,142 module.py:1284 INFO] -            ! load /home/runner/MLC/repos/local/cache/get-python3_ef885f2b/mlc-cached-state.json
  [2025-06-16 16:16:20,143 module.py:2240 INFO] - Path to Python: /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3
  [2025-06-16 16:16:20,143 module.py:2240 INFO] - Python version: 3.12.10
  [2025-06-16 16:16:20,143 module.py:1284 INFO] -          ! load /home/runner/MLC/repos/local/cache/get-generic-python-lib_5c7d1922/mlc-cached-state.json
  [2025-06-16 16:16:20,148 module.py:4013 INFO] -     - Searching for versions:  >= 1.0.0
  [2025-06-16 16:16:20,149 module.py:5124 INFO] -              ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_0bd44e38
  [2025-06-16 16:16:20,149 module.py:5125 INFO] -              ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:20,221 module.py:5270 INFO] -              ! call "detect_version" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:20,225 customize.py:117 INFO] -         Extra PIP CMD:   --break-system-packages 
  [2025-06-16 16:16:20,226 module.py:5124 INFO] -          ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_0bd44e38
  [2025-06-16 16:16:20,226 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/install.sh from tmp-run.sh
  
  /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3 -m pip install "pandas>=1.0.0" --break-system-packages
  Collecting pandas>=1.0.0
    Downloading pandas-2.3.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
  Collecting numpy>=1.26.0 (from pandas>=1.0.0)
    Downloading numpy-2.3.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (62 kB)
  Collecting python-dateutil>=2.8.2 (from pandas>=1.0.0)
    Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
  Collecting pytz>=2020.1 (from pandas>=1.0.0)
    Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
  Collecting tzdata>=2022.7 (from pandas>=1.0.0)
    Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
  Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas>=1.0.0)
    Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
  Downloading pandas-2.3.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 195.2 MB/s eta 0:00:00
  Downloading numpy-2.3.0-cp312-cp312-manylinux_2_28_x86_64.whl (16.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.6/16.6 MB 204.9 MB/s eta 0:00:00
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
  Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
  Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
  Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
  Installing collected packages: pytz, tzdata, six, numpy, python-dateutil, pandas
  
  Successfully installed numpy-2.3.0 pandas-2.3.0 python-dateutil-2.9.0.post0 pytz-2025.2 six-1.17.0 tzdata-2025.2
  [2025-06-16 16:16:28,754 module.py:5124 INFO] -          ! cd /home/runner/MLC/repos/local/cache/get-generic-python-lib_0bd44e38
  [2025-06-16 16:16:28,754 module.py:5125 INFO] -          ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/run.sh from tmp-run.sh
  [2025-06-16 16:16:28,832 module.py:5270 INFO] -          ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/customize.py
  [2025-06-16 16:16:28,836 customize.py:152 INFO] -         Detected version: 2.3.0
  [2025-06-16 16:16:28,847 module.py:2165 INFO] -     - cache UID: 0bd44e38fdee43b6
  [2025-06-16 16:16:28,856 module.py:575 INFO] -   * mlcr mlcommons,automotive,src
  [2025-06-16 16:16:28,857 module.py:1284 INFO] -        ! load /home/runner/MLC/repos/local/cache/get-mlperf-automotive-src_01435494/mlc-cached-state.json
  [2025-06-16 16:16:28,858 module.py:2240 INFO] - Path to MLPerf automotive benchmark source: /home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive
  [2025-06-16 16:16:28,861 customize.py:93 INFO] - /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3 '/home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive/tools/submission/submission_checker.py' --input 'mysubmissions/mlperf_submission' --submitter 'MLCommons' --version v0.5  
  [2025-06-16 16:16:28,861 module.py:5124 INFO] -        ! cd /home/runner/work/mlperf_automotive/mlperf_automotive
  [2025-06-16 16:16:28,861 module.py:5125 INFO] -        ! call /home/runner/MLC/repos/mlcommons@mlperf-automations/script/run-mlperf-inference-submission-checker/run.sh from tmp-run.sh
  /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3 '/home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive/tools/submission/submission_checker.py' --input 'mysubmissions/mlperf_submission' --submitter 'MLCommons' --version v0.5  
  [2025-06-16 16:16:28,918 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/deeplabv3plus/singlestream/accuracy/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,919 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,920 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,924 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/ssd/singlestream/accuracy/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,924 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/ssd/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,925 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/ssd/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,929 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/deeplabv3plus/singlestream/accuracy/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,929 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,930 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/deeplabv3plus/singlestream/performance/run_1/mlperf_log_detail.txt.
  [2025-06-16 16:16:28,930 submission_checker.py:2215 INFO] ---
  [2025-06-16 16:16:28,930 submission_checker.py:2219 INFO] Results open/MLCommons/results/pkrvmxyh4eaekms-reference-cpu-onnxruntime-default_config/deeplabv3plus/singlestream 16265.090026
  [2025-06-16 16:16:28,930 submission_checker.py:2219 INFO] Results open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/deeplabv3plus/singlestream 598.041777
  [2025-06-16 16:16:28,930 submission_checker.py:2219 INFO] Results open/MLCommons/results/pkrvmxyh4eaekms-reference-gpu-pytorch-cu124/ssd/singlestream 581.485296
  [2025-06-16 16:16:28,930 submission_checker.py:2221 INFO] ---
  [2025-06-16 16:16:28,931 submission_checker.py:2310 INFO] ---
  [2025-06-16 16:16:28,931 submission_checker.py:2311 INFO] Results=3, NoResults=0, Power Results=0
  [2025-06-16 16:16:28,931 submission_checker.py:2318 INFO] ---
  [2025-06-16 16:16:28,931 submission_checker.py:2319 INFO] Closed Results=0, Closed Power Results=0
  
  [2025-06-16 16:16:28,931 submission_checker.py:2324 INFO] Open Results=3, Open Power Results=0
  
  [2025-06-16 16:16:28,931 submission_checker.py:2329 INFO] Network Results=0, Network Power Results=0
  
  [2025-06-16 16:16:28,931 submission_checker.py:2334 INFO] ---
  [2025-06-16 16:16:28,931 submission_checker.py:2336 INFO] Systems=2, Power Systems=0
  [2025-06-16 16:16:28,931 submission_checker.py:2340 INFO] Closed Systems=0, Closed Power Systems=0
  [2025-06-16 16:16:28,931 submission_checker.py:2345 INFO] Open Systems=2, Open Power Systems=0
  [2025-06-16 16:16:28,931 submission_checker.py:2350 INFO] Network Systems=0, Network Power Systems=0
  [2025-06-16 16:16:28,931 submission_checker.py:2355 INFO] ---
  [2025-06-16 16:16:28,931 submission_checker.py:2360 INFO] SUMMARY: submission looks OK
  /opt/hostedtoolcache/Python/3.12.10/x64/bin/python3 '/home/runner/MLC/repos/local/cache/get-git-repo_automotive-src_4c2f5d09/automotive/tools/submission/generate_final_report.py' --input summary.csv  --version 0.5 
  =========================================================
  Searching for summary.csv ...
  Converting to json ...
  
                                                                             0  ...                                                  2
  Organization                                                       MLCommons  ...                                          MLCommons
  Availability                                                       available  ...                                          available
  Division                                                                open  ...                                               open
  SystemType                                                              edge  ...                                               edge
  SystemName                                                      792f6f734930  ...                                       e094ed31695a
  Platform                   pkrvmxyh4eaekms-reference-cpu-onnxruntime-defa...  ...        pkrvmxyh4eaekms-reference-gpu-pytorch-cu124
  Model                                                          deeplabv3plus  ...                                      deeplabv3plus
  MlperfModel                                                    deeplabv3plus  ...                                      deeplabv3plus
  Scenario                                                        SingleStream  ...                                       SingleStream
  Result                                                          16265.090026  ...                                         598.041777
  Accuracy                                                      mIOU: 0.924355  ...                                     mIOU: 0.924357
  number_of_nodes                                                            1  ...                                                  1
  host_processor_model_name                     Intel(R) Xeon(R) CPU @ 2.20GHz  ...                     Intel(R) Xeon(R) CPU @ 2.20GHz
  host_processors_per_node                                                   1  ...                                                  1
  host_processor_core_count                                                 16  ...                                                 16
  accelerator_model_name                                                   NaN  ...                                          NVIDIA L4
  accelerators_per_node                                                      0  ...                                                  1
  Location                   open/MLCommons/results/pkrvmxyh4eaekms-referen...  ...  open/MLCommons/results/pkrvmxyh4eaekms-referen...
  framework                                                        onnxruntime  ...                                            pytorch
  operating_system               Ubuntu 22.04 (linux-6.8.0-1030-gcp-glibc2.35)  ...      Ubuntu 22.04 (linux-6.8.0-1030-gcp-glibc2.35)
  notes                                                                    NaN  ...                                                NaN
  compliance                                                                 1  ...                                                  1
  errors                                                                     0  ...                                                  0
  version                                                                 v0.5  ...                                               v0.5
  inferred                                                                   0  ...                                                  0
  has_power                                                              False  ...                                              False
  Units                                                           Latency (ms)  ...                                       Latency (ms)
  weight_data_types                                                       fp32  ...                                               fp32
  
  [28 rows x 3 columns]
  
  =========================================================
  [2025-06-16 16:16:30,282 module.py:5270 INFO] -        ! call "postprocess" from /home/runner/MLC/repos/mlcommons@mlperf-automations/script/run-mlperf-inference-submission-checker/customize.py
```
</details>


GitHub action workflow file for submission generation through MLCFlow Automation Framework could be found [here](https://github.com/mlcommons/mlperf-automations/blob/dev/.github/workflows/test-mlperf-automotive-submission-generation.yml)