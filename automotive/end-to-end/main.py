import argparse
import pickle
import array
import os
import sys
import mlperf_loadgen as lg

from dataset import Nuscenes
from backend_deploy import BackendUniAD

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "nuscenes",
        "backend": "pytorch",
        "model-name": "uniad",
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "ConstantStream": lg.TestScenario.ConstantStream,
}

def get_args():
    parser = argparse.ArgumentParser(description="UniAD MLPerf Automotive Benchmark")
    parser.add_argument("--config", help="Path to model config file", required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint file", required=True)
    parser.add_argument("--dataset-path", help="Path to NuScenes dataset", required=True)
    parser.add_argument("--length", type=int, default=0, help="Benchmark dataset size")
    parser.add_argument(
        "--scenario",
        default="SingleStream",
        help="mlperf benchmark scenario, one of " +
        str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument("--log-dir", type=str, default="results", help="Directory to store LoadGen logs")
    parser.add_argument("--accuracy", action="store_true", help="Run benchmark in accuracy mode")
    parser.add_argument(
        "--audit_conf", default="audit.config", help="config for LoadGen audit settings"
    )
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    return parser.parse_args()

def main():
    args = get_args()
    
    print("Initializing UniAD PyTorch Backend...")
    backend = BackendUniAD(config_path=args.config, checkpoint_path=args.checkpoint)
    
    print("Initializing Nuscenes Dataset/QSL...")
    dataset = Nuscenes(os.path.abspath(args.dataset_path), config=args.config, length=args.length)

    # ---------- MLPerf SUT Callbacks ---------- #
    def issue_query(query_samples):
        # Retrieve the requested indices sent by loadgen
        sample_indices = [q.index for q in query_samples]
        
        # Load and collate data
        # dataset.get_samples now returns a list of BS=1 collated batches
        batch_data_list = dataset.get_samples(sample_indices)
        
        # Format predictions and send responses back to LoadGen
        responses = []
        for idx, q in enumerate(query_samples):
            # Execute model inference sequentially (BS=1)
            predictions = backend.predict(batch_data_list[idx])
            
            # We serialize predictions via pickle to simulate passing memory buffers.
            # OpenMMLab returns a list of results. For BS=1, we take predictions[0]
            prediction_bytes = pickle.dumps(predictions[0])
            response_array = array.array("B", prediction_bytes)
            bi = response_array.buffer_info()
            
            # Record completion with the query id, memory pointer, and length
            responses.append(lg.QuerySampleResponse(q.id, bi[0], bi[1]))
            
        lg.QuerySamplesComplete(responses)

    def flush_queries():
        pass # No buffering implementation required for basic execution

    # ---------- MLPerf Initialization ---------- #
    print("Constructing Query Sample Library (QSL)...")
    qsl = lg.ConstructGroupedQSL(
        dataset.get_item_count(), 
        min(dataset.get_item_count(), 100), # Performance count
        dataset.load_query_samples, 
        dataset.unload_query_samples
    )
    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)
    print("Constructing System Under Test (SUT)...")
    sut = lg.ConstructSUT(issue_query, flush_queries)
    
    # Configure LoadGen parameters
    settings = lg.TestSettings()
    settings.FromConfig(user_conf, 'uniad', args.scenario)
    scenario = SCENARIO_MAP[args.scenario]
    settings.scenario = scenario
    settings.mode = lg.TestMode.AccuracyOnly if args.accuracy else lg.TestMode.PerformanceOnly
    
    # Trigger Benchmark
    print(f"Starting MLPerf LoadGen in {args.scenario} scenario...")
    
    # Configure and create log directory
    if args.log_dir:
        log_dir = os.path.abspath(args.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        os.chdir(log_dir)
    
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_output_settings.copy_detail_to_stdout = False
    
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = False
    
    audit_config = os.path.abspath(args.audit_conf)
    lg.StartTestWithGroupedQSL(sut, qsl, settings, audit_config)
    
    print(f"Benchmark complete. Logs saved to {args.log_dir}/")
    print("Destroying LoadGen objects...")
    lg.DestroyGroupedQSL(qsl)
    lg.DestroySUT(sut)

if __name__ == "__main__":
    main()