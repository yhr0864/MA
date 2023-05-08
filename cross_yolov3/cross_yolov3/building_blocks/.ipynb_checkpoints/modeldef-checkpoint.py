# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# please, end the file with '}' and nothing else. this file updated automatically

MODEL_ARCH = {
    
}

# example for testing
Test_model_arch = {
    "test_net": {
    "block_op_type_backbone": [
            ["CBL_k3"], ["CBL_k3"], ["CBL_k3"],
            ["CBL_k3"], ["CBL_k3"], ["CBL_k3"]
    ],
    "block_op_type_head26": [
            ["CBL_k3"]
    ],
    "block_op_type_head13": [
            ["CBL_k3"]
    ]
    },
   
}