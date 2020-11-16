#!/usr/bin/env python3.6

import os
import main
from args import parse_args, update_from_config

maml_base = {
    'model_name': 'ours',  # seems equiv to MAML?
    'num_ways': 5,
    'num_shots': 3,
    'gamma': 0.8,  # tbd
}

NAME_TO_ARGS = {
    'default': {
        'wandb': 'cs330_finalproject_finalreport',
        'prob_statio': 0.9,
        'cl_strategy': 'loss',
        # 'cl_strategy_thres': 4.0,  # used within g_lambda (but neq lamabda itself)
        'gamma': -1,
        'n_runs': 4,
        'num_ways': 5,
        'num_shots': 5,
        'verbose': True},

    'xp_protomaml_diffN_alg4_always_update_slow_weight': {
        'model_name': 'protomaml',
        'num_ways': 5,
        'num_shots': 3,
        'use_different_nways': True,
        'cl_strategy': None,
        'num_epochs': 1,
        'gamma': -1,
        'um_power': 0.0},

#    'maml_alg4_no_um': {
#        **maml_base,
#        'um_power': 0.0},
#    'maml_alg4_no_um_G2': {
#        **maml_base,
#        'um_power': 0.0,
#        'gamma': 2.0},
#    'maml_alg4_no_um_G02': {
#        **maml_base,
#        'um_power': 0.0,
#        'gamma': 0.2},
#    'maml_alg4_yes_um': {
#        **maml_base,
#        'um_power': 1.0},
#    # BEGIN: SUNDAY
#    'maml_diffN_yes_um_G3': {
#        **maml_base,
#        'use_different_nways': True,
#        'um_power': 1.0,
#        'gamma': 3.0},
#    'maml_sameN_yes_um_G3': {
#        **maml_base,
#        'um_power': 1.0,
#        'gamma': 3.0},
#    'maml_diffN_yes_um_G03': {
#        **maml_base,
#        'use_different_nways': True,
#        'um_power': 1.0,
#        'gamma': 0.3},
#    'maml_sameN_yes_um_G03': {
#        **maml_base,
#        'um_power': 1.0,
#        'gamma': 0.3},
#    'maml_diffN_no_um_G3': {
#        **maml_base,
#        'use_different_nways': True,
#        'um_power': 0.0,
#        'gamma': 3.0},
#    'maml_sameN_no_um_G3': {
#        **maml_base,
#        'um_power': 0.0,
#        'gamma': 3.0},
#    'maml_diffN_no_um_G03': {
#        **maml_base,
#        'use_different_nways': True,
#        'um_power': 0.0,
#        'gamma': 0.3},
#    'maml_sameN_no_um_G03': {
#        **maml_base,
#        'um_power': 0.0,
#        'gamma': 0.3},
#    # END: SUNDAY
#    'maml_alg3_no_um': {
#        **maml_base,
#        'um_power': 0.0,
#        'algo3': True},
#    'maml_alg3_yes_um': {
#        **maml_base,
#        'um_power': 1.0,
#        'algo3': True},
#    'protomaml_alg4_no_um': {
#        **maml_base,
#        'model_name': 'protomaml',
#        'um_power': 0.0},
#    'protomaml_alg4_no_um_G2': {
#        **maml_base,
#        'model_name': 'protomaml',
#        'um_power': 0.0,
#        'gamma': 2.0},
#    'protomaml_alg4_yes_um': {
#        **maml_base,
#        'model_name': 'protomaml',
#        'um_power': 1.0},
#    'protomaml_alg3_no_um': {
#        **maml_base,
#        'model_name': 'protomaml',
#        'um_power': 0.0,
#        'algo3': True},
#    'protomaml_alg3_yes_um': {
#        **maml_base,
#        'model_name': 'protomaml',
#        'um_power': 1.0,
#        'algo3': True},
#    
}

# NAME = os.getenv('RUN_NAME', 'no_pretrain')
# print(f'NAME={NAME}')

if __name__ == '__main__':
    args = parse_args()

    # gpu_id = args.gpu
    # print(f'SETTING GPU TO {gpu_id}')
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'

    NAME = args.name
    overrides = {**NAME_TO_ARGS['default'], **NAME_TO_ARGS[NAME]}

    # NAME += f'_G{args.gamma}'
    NAME += f'_E{args.num_epochs}'
    if args.deeper > 0:
        NAME += f'_L{args.deeper}'
    if args.hidden_size != 64:
        NAME += f'_H{args.hidden_size}'
    if args.cl_strategy_thres != 0.9:
        NAME += f'_SThr{args.cl_strategy_thres}'
    NAME += f'_NW{args.num_ways}'

    print(f'NAME={NAME}')
    overrides['name'] = NAME

    for k, v in overrides.items():
        print(f"Setting args.{k} = {v}")
        setattr(args, k, v)
    update_from_config(args)

    main.main(args)
