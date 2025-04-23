import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
import argparse
import nuscenes_reader

def main():
    parser = argparse.ArgumentParser(description="Quantize an ONNX model.")
    parser.add_argument("model_fp32", type=str, help="Path to the input ONNX model.")
    parser.add_argument("model_quant", type=str, help="Path to save the quantized ONNX model.")
    parser.add_argument("--pkl_file", help="bevformer configuration file path")
    args = parser.parse_args()
    cal_reader = nuscenes_reader.Nuscenes(args.pkl_file)
    #quantized_model = quantize_dynamic(args.model_fp32, args.model_quant, weight_type=QuantType.Qfp)
    quantized_model = quantize_static(args.model_fp32, args.model_quant, cal_reader)
    
    print(f"Quantized model saved to {args.model_quant}")

if __name__ == "__main__":
    main()
