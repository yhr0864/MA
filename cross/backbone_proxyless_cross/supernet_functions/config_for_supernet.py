import numpy as np

CONFIG_SUPERNET = {
    'gpu_settings' : {
        'gpu_ids' : [0]
    },
    'lookup_table' : {
        'create_from_scratch' : False,
        'path_to_lookup_table' : './lookup_table.txt',
        'path_to_lookup_table_head26' : './lookup_table_head26.txt',
        'path_to_lookup_table_head13' : './lookup_table_head13.txt',
        'path_to_lookup_table_fpn' : './lookup_table_fpn.txt',
        'number_of_runs' : 50, # each operation run number_of_runs times and then we will take average

        'path_to_num_ops_table' : './num_ops_backbone.txt',
        'path_to_num_ops_head26' : './num_ops_head26.txt',
        'path_to_num_ops_head13' : './num_ops_head13.txt',
        'path_to_num_ops_fpn' : './num_ops_fpn.txt',
    },
    'logging' : {
        'path_to_log_file' : './supernet_functions/logs/logger/',
        'path_to_tensorboard_logs' : './supernet_functions/logs/tb'
    },
    'binary_mode' : 'two_v2',
    'dataloading' : {
        'img_size' : 416,
        'batch_size' : 16
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 0.1,
        'w_momentum' : 0.9,
        'w_weight_decay' : 1e-4,
        # Adam parameters for thetas
        'thetas_lr' : 0.1,
        'thetas_weight_decay' : 5 * 1e-4
    },
    'loss' : {
        'alpha' : 0.2, # 0.2 ,1
        'beta' : 0.6 # 0.6 , 0.2
    },
    'train_settings' : {
        'cnt_epochs' : 2000, # 90
        'train_thetas_from_the_epoch' : 20,
        'print_freq' : 100,
        'path_to_save_model' : './checkpoints/best_model.pth/best_model.pth',
        'path_to_save_current_model' : './checkpoints/current_model.pth/current_model.pth'
    },
    'valid_settings' : {
        'iou_thres' : 0.5,
        'conf_thres' : 0.1,
        'nms_thres' : 0.5
    },
    'quan' : {
        'act' : {
            'mode': 'lsq',
            # Bit width of quantized activation
            'bit': 8,
            # Each output channel uses its own scaling factor
            'per_channel': False,
            # Whether to use symmetric quantization
            'symmetric': False,
            # Quantize all the numbers to non-negative
            'all_positive': True
        },
        'weight' : {
            'mode': 'lsq',
            # Bit width of quantized activation
            'bit': 8,
            # Each output channel uses its own scaling factor
            'per_channel': True,
            # Whether to use symmetric quantization
            'symmetric': False,
            # Quantize all the numbers to non-negative
            'all_positive': False
        },
        'excepts' : {
            # Specify quantized bit width for some layers, like this:
            'conv1' : {
                'weight' : {'bit' : None},
                'act' : {'all_positive' : False}
            }
        }
    },
    'prune' : {
        'sr' : True,
        'scale_sparse_rate' : 5e-5
    }
}