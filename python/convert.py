import argparse
from onnx2tf import convert

def convert_onnx_to_litert(onnx_path, output_dir):
    # Write code to call convert function
    convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
        not_use_onnxsim=True,
        not_use_opname_auto_generate=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .onnx ONNX model to .tflite.")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="./models/simple_classifier.onnx",
        help="Path to the input .onnx model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/litert",
        help="Directory to save LiteRT model",
    )
    args = parser.parse_args()

    convert_onnx_to_litert(args.onnx_path, args.output_dir)
