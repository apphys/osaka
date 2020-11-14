#!/usr/bin/env python3.6

import main
from args import parse_args

maml_base = {
    'model_name': 'ours',  # seems equiv to MAML?
    'num_ways': 10,
    'num_shots': 3,
    'gamma': 0.8,  # tbd
    'cl_strategy_thres': 0.9,
}

NAME_TO_ARGS = {
    'default': {
        'wandb': 'cs330_finalproject_finalreport',
        'prob_statio': 0.9,
        'cl_strategy': 'loss',
        'cl_strategy_thres': 4.0,  # used within g_lambda (but neq lamabda itself)
        'gamma': -1,
        'n_runs': 4,
        'num_ways': 5,
        'num_shots': 5,
        'verbose': True},
    'maml_alg4_no_um': {
        **maml_base,
        'um_power': 0.0},
    'maml_alg4_yes_um': {
        **maml_base,
        'um_power': 1.0},
    'maml_alg3_no_um': {
        **maml_base,
        'um_power': 0.0,
        'algo3': True},
    'maml_alg3_yes_um': {
        **maml_base,
        'um_power': 1.0,
        'algo3': True},
    'protomaml_alg4_no_um': {
        **maml_base,
        'model_name': 'protomaml',
        'um_power': 0.0},
    'protomaml_alg4_yes_um': {
        **maml_base,
        'model_name': 'protomaml',
        'um_power': 1.0},
    'protomaml_alg3_no_um': {
        **maml_base,
        'model_name': 'protomaml',
        'um_power': 0.0,
        'algo3': True},
    'protomaml_alg3_yes_um': {
        **maml_base,
        'model_name': 'protomaml',
        'um_power': 1.0,
        'algo3': True},
}

# import os
# NAME = os.getenv('RUN_NAME', 'no_pretrain')
# print(f'NAME={NAME}')

if __name__ == '__main__':
    args = parse_args()
    NAME = args.name
    overrides = {**NAME_TO_ARGS['default'], **NAME_TO_ARGS[NAME]}

    NAME += f'_E{args.num_epochs}'
    print(f'NAME={NAME}')
    overrides['name'] = NAME

    for k, v in overrides.items():
        print(f"Setting args.{k} = {v}")
        setattr(args, k, v)
    main.main(args)
