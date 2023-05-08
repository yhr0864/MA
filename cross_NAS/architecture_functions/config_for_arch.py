CONFIG_ARCH = {
    'logging' : {
        'path_to_log_file' : './architecture_functions/logs/logger/',
        'path_to_tensorboard_logs' : './architecture_functions/logs/tb'
    },
    'dataloading' : {
        'img_size' : 416,
        'batch_size' : 32
    },
    'sub-model-saving': './sampled_model.pth',
    'pruned-model-saving': './pruned_model.pth',
    'optimizer' : {
        'lr' : 0.01,
        'momentum' : 0.9,
        'weight_decay' : 5e-4
    },
    'train_settings' : {
        'print_freq' : 50, # show logging information
        'path_to_save_model' : './checkpoints_pruned_model/best_pruned_model.pth/best_pruned_model.pth',
        'path_to_save_current_model' : './checkpoints_pruned_model/current_pruned_model.pth/current_pruned_model.pth',
        'cnt_epochs' : 200, #
        # YOU COULD USE 'CosineAnnealingLR' or 'MultiStepLR' scheduler
        'scheduler' : 'CosineAnnealingLR',
        ## CosineAnnealingLR settings
        'eta_min' : 0.001,
        ## MultiStepLR settings
        'milestones' : [90, 110], # [90, 180, 270], # decay 10x at 90, 180, and 270 epochs
        'lr_decay' : 0.1
    }
}