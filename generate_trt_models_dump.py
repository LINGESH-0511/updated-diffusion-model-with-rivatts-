#!/usr/bin/env python3
################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
################################################################################
""" Generate trt models if not present already """
import os
import argparse
import yaml
import sys
from deepmerge import always_merger

PATH_DIR_TRT = "/tmp/a2x"
PATH_A2F_TRT = "/tmp/a2x/{A2F_MODEL_NAME}.trt"
PATH_A2F_ONNX = "/data/a2f/nets_fullface/{A2F_MODEL_NAME}/network.onnx"
PATH_A2E_TRT = "/tmp/a2x/a2e.trt"
PATH_A2E_ONNX = "/data/a2e/nets/a2e_v1.4.2/network.onnx"
A2F_MODEL_LIST = ["claire_v2.3", "mark_v2.3", "james_v2.3"]


def gen_a2f_model(model_name, use_fp16, min_shape, opt_shape,max_shape):
    fullpath_trt = PATH_A2F_TRT.format(A2F_MODEL_NAME=model_name)
    if os.path.exists(fullpath_trt):
        print("A2F TRT model already exists!")
        return 0 # return 0 to indicate success

    if not os.path.exists(PATH_DIR_TRT):
        os.makedirs(PATH_DIR_TRT)

    fullpath_onnx = PATH_A2F_ONNX.format(A2F_MODEL_NAME=model_name)

    cmd = (f"trtexec "
           f"--onnx={fullpath_onnx}"
           f" --saveEngine={fullpath_trt} "
           f"--device=0 "
           f"--minShapes=input:{min_shape}x1x8320,emotion:{min_shape}x1x26 "
           f"--maxShapes=input:{max_shape}x1x8320,emotion:{max_shape}x1x26 "
           f"--optShapes=input:{opt_shape}x1x8320,emotion:{opt_shape}x1x26 ")

    if use_fp16:
        cmd+= (f"--fp16 "
           f"--precisionConstraints=prefer "
           f"--layerPrecisions='*':fp16")

    print("Generating A2F Model...")
    print(f"$ {cmd}")

    return os.system(cmd)


def gen_a2e_model( min_shape, opt_shape,max_shape):
    if os.path.exists(PATH_A2E_TRT):
        print("A2E TRT model already exists!")
        return 0 # return 0 to indicate success

    if not os.path.exists(PATH_DIR_TRT):
        os.makedirs(PATH_DIR_TRT)

    cmd = (f"trtexec "
           f"--onnx={PATH_A2E_ONNX} "
           f"--saveEngine={PATH_A2E_TRT} "
           f"--tacticSources=-CUDNN "
           f"--minShapes=input_values:{min_shape}x30000 "
           f"--maxShapes=input_values:{max_shape}x30000 "
           f"--optShapes=input_values:{opt_shape}x30000 ")

    print("Generating A2E Model...")
    print(f"$ {cmd}")

    return os.system(cmd)

def parse_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None

class ConfigCommonData:
    def __init__(self, data_stylization, data_deployment, data_advanced):


        self.a2f_model = data_stylization["a2f"]["inference_model_id"]
        self.a2f_use_fp16 =  data_advanced["trt_model_generation"]["a2f"]["precision"] == "fp16"
        self.a2f_min_shape = data_advanced["trt_model_generation"]["a2f"]["min_shape"]
        self.a2f_opt_shape = data_advanced["trt_model_generation"]["a2f"]["optimal_shape"]
        self.a2f_max_shape = data_advanced["trt_model_generation"]["a2f"]["maximum_shape"]
        self.a2e_min_shape = data_advanced["trt_model_generation"]["a2e"]["min_shape"]
        self.a2e_opt_shape = data_advanced["trt_model_generation"]["a2e"]["optimal_shape"]
        self.a2e_max_shape = data_advanced["trt_model_generation"]["a2e"]["maximum_shape"]

    def __str__(self):
        acc = ""
        acc += f"a2f_model:     {self.a2f_model}\n"
        acc += f"a2f_use_fp16:  {self.a2f_use_fp16}\n"
        acc += f"a2f_min_shape: {self.a2f_min_shape}\n"
        acc += f"a2f_opt_shape: {self.a2f_opt_shape}\n"
        acc += f"a2f_max_shape: {self.a2f_max_shape}\n"
        acc += f"a2e_min_shape: {self.a2e_min_shape}\n"
        acc += f"a2e_opt_shape: {self.a2e_opt_shape}\n"
        acc += f"a2e_max_shape: {self.a2e_max_shape}\n"
        return acc


def get_data_after_patch(default_file, patch_file):
    with open(default_file, 'r') as file:
        default_data = yaml.safe_load(file)

    if patch_file is not None:
        with open(patch_file, 'r') as file:
            patch_data = yaml.safe_load(file)
        if patch_data:
            print(f"Applying patch: {patch_data}")
            default_data = always_merger.merge(default_data, patch_data)

    return default_data



def main():
    parser = argparse.ArgumentParser(description='Generates TRT models for the A2F Service.')
    parser.add_argument('--stylization-config', type=str, help='File path to the override stylization config')
    parser.add_argument('--deployment-config', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--advanced-config', type=str, help='File path to the override advanced config')
    parser.add_argument('--default-stylization', type=str,
                        help=argparse.SUPPRESS,
                        default="/apps/configs/stylization_config.yaml", )
    parser.add_argument('--default-deployment', type=str,
                        help=argparse.SUPPRESS,
                        default="/apps/configs/deployment_config.yaml")
    parser.add_argument('--default-advanced', type=str,
                        help=argparse.SUPPRESS,
                        default="/apps/configs/advanced_config.yaml")
    args = parser.parse_args()

    print(f"Specified arguments are:")
    for key, val in vars(args).items():
        print(f"{key}={val}")

    data_stylization = get_data_after_patch(args.default_stylization, args.stylization_config)
    data_deployment =  get_data_after_patch(args.default_deployment, args.deployment_config)
    data_advanced =   get_data_after_patch(args.default_advanced, args.advanced_config)

    print("\nValues are:")
    cfg_common = ConfigCommonData(data_stylization, data_deployment, data_advanced)
    print(cfg_common)

    print("Running model generation script...")

    if cfg_common.a2f_model not in A2F_MODEL_LIST:
        raise Exception(f"Invalid A2F model {cfg_common.a2f_model} expected one of {A2F_MODEL_LIST}!")


    a2f_result = gen_a2f_model(
        cfg_common.a2f_model,
        cfg_common.a2f_use_fp16,
        cfg_common.a2f_min_shape,
        cfg_common.a2f_opt_shape,
        cfg_common.a2f_max_shape
    )

    a2e_result = gen_a2e_model(
        cfg_common.a2e_min_shape,
        cfg_common.a2e_opt_shape,
        cfg_common.a2e_max_shape
    )

    print("Model generation done.")
    if a2f_result != 0 or a2e_result != 0:
        print("Engine generation failed.")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
