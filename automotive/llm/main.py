import argparse
import array
import threading
import os
import mlperf_loadgen as lg

# Import separated modules
from dataset import MMLU_QSL
from backend_deploy import LlamaBackend

class LlamaSUT:
    def __init__(self, model_path, qsl, device="cuda"):
        self.backend = LlamaBackend(model_path, device)
        self.qsl = qsl

    def issue_queries(self, query_samples):
        threading.Thread(target=self._process_queries, args=(query_samples,)).start()

    def _process_queries(self, query_samples):
        for q in query_samples:
            # 1. Get Data
            sample_input = self.qsl.qsl_lookup[q.index]
            messages = sample_input["messages"]
            
            # 2. Run Inference
            # Returns a string (e.g., "A" or "A. The answer is...")
            pred_text = self.backend.predict(messages)

            # 3. Process Response
            # We extract the first character (A, B, C, D) 
            if pred_text and len(pred_text) > 0:
                clean_pred = pred_text.strip()[0].upper()
            else:
                clean_pred = "?"
            
            # 4. Serialize for LoadGen
            # Simple utf-8 encoding of the single character
            response_bytes = clean_pred.encode("utf-8")
            
            response_array = array.array("B", response_bytes)
            response_info = response_array.buffer_info()
            
            lg.QuerySamplesComplete([
                lg.QuerySampleResponse(q.id, response_info[0], response_info[1])
            ])

    def flush_queries(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="meta-llama/Llama-3.2-3B-Instruct", help="HF Repo ID or local path")
    parser.add_argument("--dataset_path", default="mmlu_automotive.json", help="Path to generated dataset")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--performance_sample_count", type=int, default=100, help="Number of samples to run")
    parser.add_argument("--time", type=int, default=10, help="Benchmark duration in minutes")
    parser.add_argument("--accuracy", action="store_true", help="Run in Accuracy Mode")
    parser.add_argument("--enable_trace", action="store_true", help="Enable LoadGen tracing")
    args = parser.parse_args()

    # 1. Setup QSL
    qsl = MMLU_QSL(dataset_path=args.dataset_path)
    count = min(args.performance_sample_count, qsl.count)

    # 2. Setup SUT
    sut_wrapper = LlamaSUT(args.model_path, qsl, args.device)
    
    # 3. Configure LoadGen Settings
    settings = lg.TestSettings()
    
    settings.scenario = lg.TestScenario.SingleStream
    settings.single_stream_expected_latency_ns = 1000000000 # 1 second target latency constraint
    
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    settings.min_duration_ms = args.time * 60 * 1000
    settings.min_query_count = count
    
    # Logging
    os.makedirs("results", exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = "results"
    log_output_settings.copy_summary_to_stdout = True
    
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_trace

    print(f"Starting MLPerf Benchmark: SingleStream ({'Accuracy' if args.accuracy else 'Performance'}) ...")

    # 4. Start Test
    sut = lg.ConstructSUT(sut_wrapper.issue_queries, sut_wrapper.flush_queries)
    qsl_obj = lg.ConstructQSL(
        qsl.count, 
        count, 
        qsl.load_query_samples, 
        qsl.unload_query_samples
    )

    lg.StartTestWithLogSettings(sut, qsl_obj, settings, log_settings)

    print("Benchmark Complete.")
    
    lg.DestroyQSL(qsl_obj)
    lg.DestroySUT(sut)

if __name__ == "__main__":
    main()