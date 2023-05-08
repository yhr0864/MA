# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# please, end the file with '}' and nothing else. this file updated automatically

MODEL_ARCH = {
  

    "best": {
            "block_op_type_backbone": [
            ["CBL_k7"], ["skip"], ["CBL_k5"], 
            ["CBL_k3"], ["CBL_k7"], ["CBL_k3"], 
            ],
"block_op_type_head26": [
            ["CBL_k7"], 
                ],
"block_op_type_head13": [
            ["CBL_k5"], 
                    ],
},
}