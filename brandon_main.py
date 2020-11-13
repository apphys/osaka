#!/usr/bin/env python3.6

import main
from args import parse_args

# milestone report
# MAML always update slow weight, not good. fig.1 of milstone report
#python main.py -v --prob_statio 0.9 --num_epochs 1 --num_ways 10 --num_shots=3 --num_epochs 1 --n_runs 1
# Proto-MAML always update slow weight, better than MAML. Because has better initializatoin of the last FC layer than just grad descent. good initialization before gradient descent
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 10 --num_shots=3 --n_runs 1
# Proto-MAML always update slow weight.  pretty good since only 5 ways.
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --n_runs 1 --use_different_nway True

# final report
# MAML, Alg4, no UM
#python main.py -v --prob_statio 0.9 --num_ways 10 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 0.8 --cl_strategy_thres 0.9 --cl_strategy 'loss'
# MAML, Alg4, with UM
#python main.py -v --prob_statio 0.9 --num_ways 10 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 0.8 --cl_strategy_thres 0.9 --cl_strategy 'loss' --um_power 1.0
# MAML, Alg3, with UM
#python main.py -v --prob_statio 0.9 --num_ways 10 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 3.0 --cl_strategy_thres 5.0 --cl_strategy 'loss' --um_power 1.0 --algo3 True
# Proto-MAML, Alg3, with UM
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_ways 10 --num_shots=3 --num_epochs 2 --n_runs 1 --cl_tbd_thres 2.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 True
# Proto-MAML, Alg3, with UM, with changing ways. worse than algo4
# python main.py -v --model_name protomaml --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 2.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 True --use_different_nway True
NAME_TO_ARGS = {
    'default': {
        'wandb': 'cs330_finalproject_debugging',
        'name': 'maml',
        'prob_statio': 0.9,
        'cl_strategy': 'loss',
        'cl_tbd_thresh': 0.5,
        'num_epochs': 1,
        'num_shots': 3,
        'verbose': True,
    },
    'protomaml': {
        'name': 'protomaml',
        'model_name': 'protomaml',
        'use_different_nway': True
    }
}

NAME = 'protomaml'

if __name__ == '__main__':
    args = parse_args()

    overrides = {**NAME_TO_ARGS['default'], **NAME_TO_ARGS[NAME]}
    for k, v in overrides.items():
        print(f"Setting args.{k} = {v}")
        setattr(args, k, v)
    main.main(args)
