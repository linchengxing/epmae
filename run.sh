# # pre-train
# CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/transformer.yaml --exp_name exp104_drop_path_rate_0.1_block_beta_0.5_search_size_3_patch_size_16_momentum_linear_0.9995_0.99999_mask_ratio_0.75_block_grid_size_0.02_lr_1e-4_rec_layer_9_to_12_pred_ratio_0.4
# CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/transformer.yaml --exp_name test

# # fine-tune modelnet40
# CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/transformer/modelnet40.yaml --finetune_model --exp_name exp102_2_best_warmup_10_patch_size_32_seq_len_64_drop_path_rate_0.1_lr_0.0005_mean_max_concat --ckpts experiments/pre-training/transformer/exp102_pred_ratio_0.5_drop_path_rate_0.1_block_beta_0.5_search_size_3_patch_size_16_momentum_linear_0.9995_0.99999_mask_ratio_0.75_block_grid_size_0.02_lr_1e-4_rec_layer_12/ckpt-best.pth --vote

# fine-tune scanobjectnn pb
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/transformer/scan_pb.yaml --finetune_model --exp_name exp60_29_best_warmup_10_patch_size_32_seq_len_128_max_mean_concat_drop_path_rate_0_lr_0.00005 --ckpts experiments/pre-training/transformer/exp94_pred_ratio_0.5_drop_path_rate_0.1_block_beta_0.5_lr_1e-4_warm_up_30_search_size_3_rec_layer_9_10_11_12_patch_size_16_momentum_linear_0.9995_0.99999_mask_ratio_0.75_block_grid_size_0.0001/ckpt-best.pth