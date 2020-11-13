#!/usr/bin/env python3.6

import main
from args import parse_args


NAME_TO_ARGS = {
    'default': {
        'wandb': 'cs330_finalproject_debugging',
        'name': 'maml',
        'prob_statio': 0.9,
        'cl_strategy': 'loss',
        'cl_tbd_thresh': 0.5,
        'num_epochs': 1,
        'num_shots': 3,
    }
}

NAME = 'default'

if __name__ == '__main__':
    args = parse_args()
    for k, v in NAME_TO_ARGS[NAME].items():
        setattr(args, k, v)

    print('is_classif:', args.is_classification_task)
    # main.main(args)
