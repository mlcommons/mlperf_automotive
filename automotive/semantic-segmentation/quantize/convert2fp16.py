import onnx
from onnxconverter_common import float16
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to FP16.")
    parser.add_argument("input", type=str, help="Path to input ONNX model file")
    parser.add_argument("output", type=str, help="Path to save FP16 ONNX model")
    args = parser.parse_args()

    model = onnx.load(args.input)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, args.output)

if __name__ == "__main__":
    main()
