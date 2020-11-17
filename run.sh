# MAML
# Algo4, always task shift and ood, so overfit slow-weight to current task, no UM
#python main.py -v --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1
# No UM
#python main.py -v --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss'
# With UM
#python main.py -v --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0
# Algo3, with UM
#python main.py -v --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 True



# Proto-MAML
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss'
# With UM
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 False
# chaning ways, with UM
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --use_different_nway True --um_power 1.0 --cl_strategy 'loss' --algo3 True

# milestone report
# MAML always update slow weight, not good. fig.1 of milstone report
#python main.py -v --prob_statio 0.9 --num_epochs 1 --num_ways 10 --num_shots=3 --num_epochs 1 --n_runs 1
# Proto-MAML always update slow weight, better than MAML. Because has better initializatoin of the last FC layer than just grad descent. good initialization before gradient descent
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 10 --num_shots=3 --n_runs 1 
# Proto-MAML always update slow weight.  pretty good since only 5 ways. 
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_epochs 1 --num_ways 5 --num_shots=3 --n_runs 1 --use_different_nway True

# final report
# MAML, Alg4, no UM
python main.py -v --prob_statio 0.9 --num_ways 10 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 0.8 --cl_strategy_thres 0.9 --cl_strategy 'loss'
# MAML, Alg4, with UM
#python main.py -v --prob_statio 0.9 --num_ways 10 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 0.8 --cl_strategy_thres 0.9 --cl_strategy 'loss' --um_power 1.0
# MAML, Alg3, with UM
#python main.py -v --prob_statio 0.9 --num_ways 10 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 3.0 --cl_strategy_thres 5.0 --cl_strategy 'loss' --um_power 1.0 --algo3 True
# Proto-MAML, Alg3, with UM
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_ways 10 --num_shots=3 --num_epochs 2 --n_runs 1 --cl_tbd_thres 2.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 True
# Proto-MAML, Alg3, with UM, with changing ways. worse than algo4
#python main.py -v --model_name protomaml --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 2.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 True --use_different_nway True
