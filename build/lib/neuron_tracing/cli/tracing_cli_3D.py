import argparse
import os
import platform
import sys
import json
import pkg_resources


from neuron_tracing.model.snake_3D import snake_3D


def read_parameters():
    parser = argparse.ArgumentParser(description="Snake-for-Neuron-Tracing-3D 1.0")
    parser.add_argument("--input_tiff_dir", "-A", help="input tiff file dir", required=True)
    parser.add_argument("--input_swc_dir", "-B", help="input swc file dir", required=True)
    parser.add_argument("--output_swc_dir", "-O", help="output swc file dir", required=True)
    parser.add_argument("--config", "-C", help="config for snake method", required=True)
    parser.add_argument("--show_result", "-s", help="show the optimization process", required=True)
    return parser.parse_args()

def run():
    abs_dir = os.path.abspath("")
    sys.path.append(abs_dir)
    # init path parameter
    abs_dir = os.path.abspath("")
    sys.path.append(abs_dir)
    print(abs_dir)

    args = read_parameters()

    input_image_dir = args.input_tiff_dir
    input_swc_dir = args.input_swc_dir
    output_swc_dir = args.output_swc_dir
    json_file_path = args.config
    show_result = args.show_result

    with open(json_file_path, "r") as f:
        data = json.load(f)

    snake_3D(input_image_dir, input_swc_dir, output_swc_dir, alpha=data['alpha'], beta=data['beta'], lambda_=data['lambda_'], nits=data['nits'], show_result=show_result)



if __name__ == "__main__":
    sys.exit(run())

# neurontracing3D --input_tiff_dir data/single_branch/noise_image_70/noise_image_70.tif --input_swc_dir data/single_branch/noise_image_70/noise_image_70.swc --output_swc_dir data/single_branch/noise_image_70/noise_image_70.new.swc --config config/default.json --show_result True
# python neuron_tracing/cli/tracing_cli_3D.py --input_tiff_dir data/single_branch/noise_image_70/noise_image_70.tif --input_swc_dir data/single_branch/noise_image_70/noise_image_70.swc --output_swc_dir data/single_branch/noise_image_70/noise_image_70.new.swc --config config/default.json --show_result True