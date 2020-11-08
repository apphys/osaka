# MAML
# always task shift and ood, so overfit slow-weight to current task, no UM
#python main.py -v --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1
# No UM
#python main.py -v --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss'
# With UM
#python main.py -v --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0
# Algo3, with UM
#python main.py -v --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 False


# Proto-MAML
# No UM
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss'
# With UM
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 False
# chaning ways, with UM
python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --use_different_nway True --um_power 1.0 --cl_strategy 'loss' --algo3 False

