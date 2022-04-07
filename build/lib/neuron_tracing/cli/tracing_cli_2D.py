import argparse
import os
import platform
import sys
import json
abs_dir = os.path.abspath("")
sys.path.append(abs_dir)

from neuron_tracing.model.snake_2D import snake_2D


def read_parameters():
    parser = argparse.ArgumentParser(description="Snake-for-Neuron-Tracing-2D 1.0")
    parser.add_argument("--input_tiff_dir", "-A", help="input tiff file dir", required=True)
    parser.add_argument("--init_pos_x", "-x", help="x", required=True)
    parser.add_argument("--init_pos_y", "-y", help="y", required=True)
    parser.add_argument("--init_r", "-r", help="r", required=True)
    parser.add_argument("--lambda_", "-C", help="test", required=True)
    parser.add_argument("--show_result", "-D", help="True or False", required=True)
    return parser.parse_args()

def run():
    # init path parameter
    args = read_parameters()

    input_tiff_dir = args.input_tiff_dir
    init_pos = [0, 0]
    init_pos[0] = args.init_pos_x
    init_pos[1] = args.init_pos_y
    init_r = args.init_r
    lambda_ = args.lambda_
    show_result = args.show_result

    snake_2D(input_tiff_dir, init_pos, init_r, lambda_, show_result)



if __name__ == "__main__":
    sys.exit(run())

# neurontracing2D --input_tiff_dir data/2D/test.tif --init_pos_x 10 --init_pos_y 16 --init_r 6 --lambda_ 0.3 --show_result True
# python neuron_tracing/cli/tracing_cli_2D.py --input_tiff_dir data/2D/test.tif --init_pos_x 10 --init_pos_y 16 --init_r 6 --lambda_ 0.3 --show_result True