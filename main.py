def define_env(env):

    @env.macro
    def mlperf_inference_implementation_readme(
        spaces,
        model,
        implementation,
        *,
        implementation_tips=True,
        setup_tips=True,
        run_tips=True,
        skip_test_query_count=False,
        fixed_scenarios=[],
        devices=[],
        frameworks=[],
        categories=[],
        extra_variation_tags="",
        extra_input_string="",
        extra_docker_input_string="",
    ):
        pre_space = ""

        for i in range(1, spaces):
            pre_space = pre_space + " "
        f_pre_space = pre_space
        pre_space += " "

        content = ""

        execution_envs = ["Docker", "Native"]
        code_version = "v0.5"
        implementation_run_options = []

        if implementation == "reference":
            if "99.9" not in model and implementation_tips:
                content += f"\n{pre_space}!!! tip\n\n"
                content += f"{pre_space}    - MLCommons reference implementations are only meant to provide a rules compliant reference implementation for the submitters and in most cases are not best performing. If you want to benchmark any system, it is advisable to use the vendor MLPerf implementation for that system like Nvidia, Intel etc.\n\n"

            if not devices:
                devices = ["CPU", "CUDA"]

            if not frameworks:
                if model.lower() in ["ssd", "bevformer", "deeplabv3plus"]:
                    frameworks = ["Onnxruntime", "Pytorch"]

        if not categories:
            if model.lower() in ["ssd", "bevformer", "deeplabv3plus"]:
                categories = ["Edge"]

        # model name
        content += f"{pre_space}{model.upper()}\n\n"

        final_run_mode = "valid" if "short" not in extra_variation_tags else "test"

        for category in categories:
            if category == "Edge":
                scenarios = ["SingleStream", "ConstantStream"]
                scenarios.remove("ConstantStream")  # TODO: Add ConstantStream
            if fixed_scenarios:
                scenarios = [
                    scenario for scenario in scenarios if scenario in fixed_scenarios]

            content += f"{pre_space}=== \"{category.lower()}\"\n\n"

            cur_space = pre_space + "    "
            scenarios_string = ", ".join(scenarios)

            content += f"{cur_space}### {category} category \n\n{cur_space} In the {category.lower()} category, {model} has {scenarios_string} scenarios and all the scenarios are mandatory for a closed division submission.\n\n"

            for framework in frameworks:
                cur_space1 = cur_space + "    "
                content += f'{cur_space}=== "{framework}"\n'
                content += f"{cur_space1}#### {framework} framework\n\n"

                for device in devices:
                    if device.lower() == "cuda" and framework.lower() == "onnxruntime":
                        continue

                    cur_space2 = cur_space1 + "    "
                    cur_space3 = cur_space2 + "    "
                    cur_space4 = cur_space3 + "    "

                    content += f"{cur_space1}=== \"{device}\"\n"
                    content += f"{cur_space2}##### {device} device\n\n"

                    # minimum system requirements
                    # TODO: Add minimum system requirements based on the newly
                    # prepocessed dataset
                    content += get_min_system_requirements(

                        cur_space2, model, implementation, device
                    )

                    # to select the execution environments(currently Docker and
                    # Native)
                    for execution_env in execution_envs:
                        content += f'{cur_space2}=== "{execution_env}"\n'
                        content += f"{cur_space3}###### {execution_env} Environment\n\n"
                        # ref to MLCFlow installation
                        content += f"{cur_space3}Please refer to the [installation page](site:inference/install/) to install MLCFlow for running the automated benchmark commands.\n\n"
                        test_query_count = get_test_query_count(

                            model, implementation, device.lower()
                        )
                        if (
                            device.lower() == "cuda"
                            and execution_env.lower() == "native"
                        ):
                            content += f"\n{cur_space3}!!! tip\n\n"
                            content += f"{cur_space3}    - It is advisable to use the commands in the Docker tab for CUDA. Run the below native command only if you are already on a CUDA setup with cuDNN and TensorRT installed.\n\n"

                        if (
                            "99.9" not in model
                        ):  # not showing docker command as it is already done for the 99% variant

                            setup_run_cmd = mlperf_inference_run_command(
                                spaces + 17,
                                model,
                                implementation,
                                framework.lower(),
                                category.lower(),
                                "SingleStream",
                                device.lower(),
                                "test",
                                test_query_count,
                                True,
                                skip_test_query_count,
                                scenarios,
                                code_version,
                                extra_variation_tags,
                                extra_input_string,
                                extra_docker_input_string,
                            )

                            common_info = get_common_info(
                                spaces + 16,
                                implementation,
                                model.lower()
                            )

                            if (
                                execution_env == "Native"
                            ):  # Native implementation steps through virtual environment
                                content += f"{cur_space3}####### Setup a virtual environment for Python\n"
                                content += get_venv_command(spaces + 16)
                                content += f"{cur_space3}####### Performance Estimation for Offline Scenario\n"

                                content += common_info

                                content += setup_run_cmd.replace(
                                    "--docker ", "")

                                content += f"{cur_space3}The above command should do a test run of Offline scenario and record the estimated offline_target_qps.\n\n"

                            else:  # Docker implementation steps
                                content += f"{cur_space3}####### Docker Container Build and Performance Estimation for Offline Scenario\n"
                                docker_info = get_docker_info(
                                    spaces + 16,
                                    model,
                                    implementation,
                                    device,
                                    setup_tips,
                                )

                                content += common_info

                                content += docker_info

                                content += setup_run_cmd

                                if len(scenarios) == 1:
                                    scenario_text = f"""the {scenarios[0]} scenario"""
                                else:
                                    scenario_text = "each scenario" ""
                                content += f"{cur_space3}The above command should get you to an interactive shell inside the docker container and do a quick test run for the Offline scenario. Once inside the docker container please do the below commands to do the accuracy + performance runs for {scenario_text}.\n\n"
                                content += f"{cur_space3}<details>\n"
                                content += f"{cur_space3}<summary> Please click here to see more options for the docker launch </summary>\n\n"
                                content += f"{cur_space3}* `--docker_mlc_repo=<Custom MLC GitHub repo URL in username@repo format>`: to use a custom fork of cm4mlops repository inside the docker image\n\n"
                                content += f"{cur_space3}* `--docker_mlc_repo_branch=<Custom MLC GitHub repo Branch>`: to checkout a custom branch of the cloned cm4mlops repository inside the docker image\n\n"
                                content += f"{cur_space3}* `--docker_cache=no`: to not use docker cache during the image build\n"

                                if device.lower() not in ["cuda"]:
                                    content += f"{cur_space3}* `--docker_os=ubuntu`: ubuntu and rhel are supported. \n"
                                    content += f"{cur_space3}* `--docker_os_version=20.04`: [20.04, 22.04] are supported for Ubuntu and [8, 9] for RHEL\n"

                                content += f"{cur_space3}</details>\n"
                        else:
                            content += f"{cur_space3} You can reuse the same environment as described for {model.split('.')[0]}.\n"
                            content += f"{cur_space3}###### Performance Estimation for Offline Scenario\n"

                            content += mlperf_inference_run_command(
                                spaces + 17,
                                model,
                                implementation,
                                framework.lower(),
                                category.lower(),
                                "SingleStream" if model.lower() in [
                                    "pointpainting"] else "Offline",
                                device.lower(),
                                "test",
                                test_query_count,
                                True,
                                skip_test_query_count,
                                scenarios,
                                code_version,
                            ).replace("--docker ", "")
                            content += f"{cur_space3}The above command should do a test run of Offline scenario and record the estimated offline_target_qps.\n\n"

                        run_suffix = ""
                        run_suffix += f"{cur_space3}<details>\n"
                        run_suffix += f"{cur_space3}<summary> Please click here to see more options for the RUN command</summary>\n\n"
                        run_suffix += f"{cur_space3}* Use `--division=closed` to do a closed division submission which includes compliance runs\n\n"
                        run_suffix += f"{cur_space3}* Use `--rerun` to do a rerun even when a valid run exists\n"
                        run_suffix += f"{cur_space3}* Use `--compliance` to do the compliance runs (only applicable for closed division) once the valid runs are successful\n"

                        for scenario in scenarios:
                            content += f"{cur_space3}=== \"{scenario}\"\n{cur_space4}###### {scenario}\n\n"
                            run_cmd = mlperf_inference_run_command(
                                spaces + 21,
                                model,
                                implementation,
                                framework.lower(),
                                category.lower(),
                                scenario,
                                device.lower(),
                                final_run_mode,
                                test_query_count,
                                False,
                                skip_test_query_count,
                                scenarios,
                                code_version,
                                extra_variation_tags,
                                extra_input_string,
                            )
                            content += run_cmd
                            # content += run_suffix

                        if len(scenarios) > 1:
                            content += f"{cur_space3}=== \"All Scenarios\"\n{cur_space4}###### All Scenarios\n\n"
                            run_cmd = mlperf_inference_run_command(
                                spaces + 21,
                                model,
                                implementation,
                                framework.lower(),
                                category.lower(),
                                "All Scenarios",
                                device.lower(),
                                final_run_mode,
                                test_query_count,
                                False,
                                skip_test_query_count,
                                scenarios,
                                code_version,
                                extra_variation_tags,
                                extra_input_string,
                            )
                            content += run_cmd

                        content += run_suffix

        readme_prefix = get_readme_prefix(
            spaces, model, implementation, extra_variation_tags
        )

        readme_suffix = get_readme_suffix(
            spaces, model, implementation, extra_variation_tags
        )

        return readme_prefix + content + readme_suffix

    def get_test_query_count(model, implementation, device, num_devices=1):

        p_range = 10

        if device == "cuda":
            p_range *= 5
            p_range *= num_devices

        return p_range

    def get_min_system_requirements(spaces, model, implementation, device):
        model = model.lower()
        min_sys_req_content = ""
        min_sys_req_content += f"{spaces}<details>\n"
        min_sys_req_content += f"{spaces}<summary>Please click here to see the minimum system requirements for running the benchmark</summary>\n\n"
        # device memory
        if device.lower() == "cuda" and (
            implementation.lower() == "nvidia" or implementation.lower() == "reference"
        ):
            if implementation.lower() == "nvidia":
                if "dlrm" in model:
                    device_memory = "24GB"
                elif "llama2-70b" in model or "mixtral" in model:
                    device_memory = "80GB"
                elif "sdxl" in model or "gptj" in model:
                    device_memory = "16GB"
                else:
                    device_memory = "8GB"
            elif implementation.lower() == "reference":
                if "dlrm" in model:
                    device_memory = "2x80GB"
                elif "llama2-70b" in model:
                    device_memory = "8x80GB"
                elif "mixtral" in model:
                    device_memory = "4x80GB"
                elif "sdxl" in model:
                    device_memory = "24GB(fp32), 16GB(fp16)"
                elif "gptj" in model:
                    device_memory = "80GB(fp32). 40GB(fp16)"
                elif "pointpainting" in model:
                    device_memory = "To be updated"
                else:
                    device_memory = "8GB"
            min_sys_req_content += f"{spaces}* **Device Memory**: {device_memory}\n\n"
        # disk space
        if "dlrm" in model:
            disk_space = "500GB"
        elif "llama2-70b" in model:
            disk_space = "700GB"
        elif "mixtral" in model:
            disk_space = "100GB"
        elif "retinanet" in model:
            disk_space = "200GB"
        elif "pointpainting" in model:
            disk_space = "To be updated"
        else:
            disk_space = "50GB"
        min_sys_req_content += f"{spaces}* **Disk Space**: {disk_space}\n\n"
        # System memory
        if "dlrm" in model:
            system_memory = "512GB"
            min_sys_req_content += (
                f"{spaces}* **System Memory(RAM+SWAP)**: {system_memory}\n\n"
            )
        min_sys_req_content += f"{spaces}</details>\n"
        return min_sys_req_content

    def get_inference_server_run_cmd(spaces, implementation):
        indent = " " * spaces + " "
        if implementation == "neuralmagic":
            pre_space = " " * spaces
            return f"""\n
{pre_space}```bash
{pre_space}mlcr run,vllm-server \\
{indent}--model=nm-testing/Llama-2-70b-chat-hf-FP8 \\
{indent}--vllm_model_name=nm-testing/Llama-2-70b-chat-hf-FP8 \\
{indent}--quiet
{pre_space}```\n"""

    def get_venv_command(spaces):
        pre_space = " " * spaces
        return f"""\n
{pre_space}```bash
{pre_space}mlcr install,python-venv --name=mlperf
{pre_space}export MLC_SCRIPT_EXTRA_CMD=\"--adr.python.name=mlperf\"
{pre_space}```\n"""

    # contains run command information which is common to both docker and
    # native runs
    def get_common_info(spaces, implementation, model):
        info = ""
        pre_space = ""
        for i in range(1, spaces):
            pre_space = pre_space + " "
        pre_space += " "
        # pre_space = "                "
        info += f"\n{pre_space}!!! tip\n\n"
        # tags for compliance test will be uncommented when the automotive round mandates a compliance run
        # info += f"{pre_space}    - Compliance runs can be enabled by adding `--compliance=yes`.\n\n"
        # to be uncomented after reviewing the constantstream scenario
        # info += f"{pre_space}    - Number of threads could be adjusted using `--threads=#`, where `#` is the desired number of threads. This option works only if the implementation in use supports threading.\n\n"
        # info += f"{pre_space}    - Batch size could be adjusted using
        # `--batch_size=#`, where `#` is the desired batch size. This option
        # works only if the implementation in use is supporting the given batch
        # size.\n\n"
        info += f"{pre_space}    - The maximum duration for a performance run can be disabled by using `--env.MLC_MLPERF_USE_MAX_DURATION=no`.\n\n"
        info += f"{pre_space}    - In valid execution mode, the query count for performance mode can be adjusted using `--env.MLC_MLPERF_LOADGEN_QUERY_COUNT=<query_count>`.\n\n"

        if implementation.lower() == "reference":
            info += f"{pre_space}    - Add `--adr.mlperf-implementation.tags=_branch.master,_repo.<CUSTOM_INFERENCE_REPO_LINK>` if you are modifying the official MLPerf Inference implementation in a custom fork.\n\n"
            info += f"{pre_space}    - Add `--adr.inference-src.tags=_repo.<CUSTOM_INFERENCE_REPO_LINK>` if you are modifying the model config accuracy script in the submission checker within a custom fork.\n\n"
            info += f"{pre_space}    - Add `--adr.inference-src.version=custom` if you are using the modified MLPerf Inference code or accuracy script on submission checker within a custom fork.\n\n"

        return info

    def get_docker_info(spaces, model, implementation,
                        device, setup_tips=True):
        info = ""
        pre_space = ""
        for i in range(1, spaces):
            pre_space = pre_space + " "
        pre_space += " "
        # pre_space = "                "
        if setup_tips:
            info += f"\n{pre_space}!!! tip\n\n"

            if model == "sdxl":
                info += f"{pre_space}    - `--env.MLC_MLPERF_MODEL_SDXL_DOWNLOAD_TO_HOST=yes` option can be used to download the model on the host so that it can be reused across different container lanuches. \n\n"
            elif "llama3" in model.lower():
                info += f"{pre_space}    - `--env.MLC_MLPERF_MODEL_LLAMA3_DOWNLOAD_TO_HOST=yes` option can be used to download the model on the host so that it can be reused across different container lanuches. \n\n"
                info += f"{pre_space}    - `--env.MLC_MLPERF_DATASET_LLAMA3_DOWNLOAD_TO_HOST=yes` option can be used to download the dataset on the host so that it can be reused across different container lanuches. \n\n"
            if implementation.lower() == "nvidia":
                info += f"{pre_space}    - Default batch size is assigned based on [GPU memory](https://github.com/mlcommons/cm4mlops/blob/dd0c35856969c68945524d5c80414c615f5fe42c/script/app-mlperf-inference-nvidia/_cm.yaml#L1129) or the [specified GPU](https://github.com/mlcommons/cm4mlops/blob/dd0c35856969c68945524d5c80414c615f5fe42c/script/app-mlperf-inference-nvidia/_cm.yaml#L1370). Please click more option for *docker launch* or *run command* to see how to specify the GPU name.\n\n"
                info += f"{pre_space}    - When run with `--all_models=yes`, all the benchmark models of NVIDIA implementation can be executed within the same container.\n\n"
                if "llama2" in model.lower():
                    info += f"{pre_space}    - The dataset for NVIDIA's implementation of Llama2 is not publicly available. The user must fill [this](https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform?pli=1&fbzx=-8842630989397184967) form and be verified as a MLCommons member to access the dataset.\n\n"
                    info += f"{pre_space}    - `PATH_TO_PICKE_FILE` should be replaced with path to the downloaded pickle file.\n\n"
        else:
            if model == "sdxl":
                info += f"\n{pre_space}!!! tip\n\n"
                info += f"{pre_space}    - `--env.MLC_MLPERF_MODEL_SDXL_DOWNLOAD_TO_HOST=yes` option can be used to download the model on the host so that it can be reused across different container lanuches. \n\n"

        # return empty string if nothing is filled inside the tip
        if info == f"\n{pre_space}!!! tip\n\n":
            return ""

        return info

    def get_readme_prefix(spaces, model, implementation, extra_variation_tags):
        readme_prefix = ""
        pre_space = "    "
        # for i in range(1,spaces):
        #     pre_space  = pre_space + " "
        # pre_space += "  "

        return readme_prefix

    def get_readme_suffix(spaces, model, implementation, extra_variation_tags):
        readme_suffix = ""
        pre_space = ""
        for i in range(1, spaces):
            pre_space = pre_space + " "
        pre_space += "  "

        if implementation == "reference" and not extra_variation_tags:
            if not model.endswith("-99"):
                model_base_name = model.replace("-99.9", "").replace("-99", "")
                readme_suffix += f"{pre_space}* If you want to download the official MLPerf model and dataset for {model} you can follow [this README](get-{model_base_name}-data.md).\n"
        return readme_suffix

    def get_run_cmd_extra(
        f_pre_space,
        model,
        implementation,
        device,
        scenario,
        scenarios=[],
        run_tips=True,
        extra_input_string="",
    ):
        extra_content = ""
        f_pre_space += ""
        # TBD: After testing the constantstream scenario, we can uncomment this
        # if scenario == "Server" or (
        #     scenario == "All Scenarios" and "Server" in scenarios
        # ):
        #     extra_content += f"{f_pre_space}    * `<SERVER_TARGET_QPS>` must be determined manually. It is usually around 80% of the Offline QPS, but on some systems, it can drop below 50%. If a higher value is specified, the latency constraint will not be met, and the run will be considered invalid.\n"
        if extra_content:
            extra_content = f"{f_pre_space}!!! tip\n\n" + extra_content

        if run_tips:
            return extra_content
        else:
            return ""

    @env.macro
    def mlperf_inference_run_command(
        spaces,
        model,
        implementation,
        framework,
        category,
        scenario,
        device="cpu",
        execution_mode="test",
        test_query_count="20",
        docker=False,
        skip_test_query_count=False,
        scenarios=[],
        code_version="r4.1-dev",
        extra_variation_tags="",
        extra_input_string="",
        extra_docker_input_string="",
    ):
        pre_space = ""
        for i in range(1, spaces):
            pre_space = pre_space + " "
        f_pre_space = pre_space
        pre_space += "  "

        if scenario == "All Scenarios":
            scenario_variation_tag = ",_all-scenarios"
            scenario_option = ""
        else:
            scenario_variation_tag = ""
            scenario_option = f"\\\n{pre_space} --scenario={scenario}"

        # TBD: After testing the constantstream scenario, we can uncomment this
        # if scenario == "Server" or (
        #     scenario == "All Scenarios" and "Server" in scenarios
        # ):
        #     scenario_option += (
        #         f"\\\n{pre_space} --server_target_qps=<SERVER_TARGET_QPS>"
        #     )

        run_cmd_extra = get_run_cmd_extra(
            f_pre_space,
            model,
            implementation,
            device,
            scenario,
            scenarios,
            True,
            extra_input_string,
        )

        if docker:
            docker_cmd_suffix = f" \\\n{pre_space} --docker --quiet"
            if not skip_test_query_count:
                docker_cmd_suffix += (
                    f" \\\n{pre_space} --test_query_count={test_query_count}"
                )
            if extra_docker_input_string != "" or extra_input_string != "":
                docker_cmd_suffix += (
                    f" \\\n{pre_space} {extra_docker_input_string} {extra_input_string}"
                )

            if "short" in extra_variation_tags:
                full_ds_needed_tag = ""
            else:
                full_ds_needed_tag = "_full,"

            docker_setup_cmd = f"""\n
{f_pre_space}```bash
{f_pre_space}mlcr run-abtf-inference,reference,_find-performance,{full_ds_needed_tag}_{code_version}{scenario_variation_tag}{extra_variation_tags} \\
{pre_space} --model={model} \\
{pre_space} --implementation={implementation} \\
{pre_space} --framework={framework} \\
{pre_space} --category={category} {scenario_option} \\
{pre_space} --execution_mode=test \\
{pre_space} --device={device} {docker_cmd_suffix}
{f_pre_space}```\n"""

            return docker_setup_cmd + run_cmd_extra

        else:
            cmd_suffix = f"\\\n{pre_space} --quiet {extra_input_string}"

            if execution_mode == "test" and not skip_test_query_count:
                cmd_suffix += f" \\\n {pre_space} --test_query_count={test_query_count}"

            if "short" in extra_variation_tags:
                full_ds_needed_tag = ""
            else:
                full_ds_needed_tag = "_full,"

            run_cmd = f"""\n
{f_pre_space}```bash
{f_pre_space}mlcr run-abtf-inference,reference,{full_ds_needed_tag}_{code_version}{scenario_variation_tag}{extra_variation_tags} \\
{pre_space} --model={model} \\
{pre_space} --implementation={implementation} \\
{pre_space} --framework={framework} \\
{pre_space} --category={category} {scenario_option} \\
{pre_space} --execution_mode={execution_mode} \\
{pre_space} --device={device} {cmd_suffix}
{f_pre_space}```\n"""

            return run_cmd + run_cmd_extra
