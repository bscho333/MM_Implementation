import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Full inference script")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./input/test/image",
        help="Path to input directory",
    )

    args = parser.parse_args()
    return args
def main():
    args = parse_args()

    # Todo: Dataset Prepare

    # Todo: Model Prepare

    # Todo: Model Training

    # Todo: extra work


if __name__ == '__main__':
    main()