import onnx
import onnxruntime.quantization as quantization
import argparse
import cognata_reader
import csv


def read_dataset_csv(file_path):
    files = []
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            files.append(row)
    return files


def main():
    parser = argparse.ArgumentParser(description="Quantize an ONNX model.")
    parser.add_argument(
        "model_fp32",
        type=str,
        help="Path to the input ONNX model.")
    parser.add_argument(
        "model_prep_path",
        type=str,
        help="Path to the prepped ONNX model.")
    parser.add_argument(
        "model_quant",
        type=str,
        help="Path to save the quantized ONNX model.")
    parser.add_argument(
        "--data-root",
        type=str,
        help="path to calibration files")
    args = parser.parse_args()
    files = read_dataset_csv("calibration_set.csv")
    cal_reader = cognata_reader.Cognata(args.data_root, len(files))
    # quantized_model = quantize_dynamic(args.model_fp32, args.model_quant, weight_type=QuantType.Qfp)
    quantization.shape_inference.quant_pre_process(
        args.model_fp32, args.model_prep_path)
    quantized_model = quantization.quantize_static(
        args.model_prep_path, args.model_quant, cal_reader)

    print(f"Quantized model saved to {args.model_quant}")


if __name__ == "__main__":
    main()
