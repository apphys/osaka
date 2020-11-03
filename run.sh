#--prob_statio
#--cl_strategy 'loss'
#--cl_strategy_thres 0.1  # determine ood
#--cl_tbd_thres -1 (task boundary, loss change)

# Proto-MAML
#python main.py -v --prob_statio 0.9 --cl_strategy loss --cl_tbd_thres 0.9 --num_epochs 5 --num_ways 10 --num_shots=3 --model_name protomaml #--use_different_nway True
#python main.py -v --prob_statio 0.9 --num_epochs 1 --num_ways 10 --num_shots=3 --num_epochs 10 --n_runs 10 --model_name protomaml --use_different_nway True
python main.py -v --prob_statio 0.9 --num_epochs 5 --num_ways 5 --num_shots=3 --n_runs 5 --model_name protomaml --use_different_nway True
# MAML
#python main.py -v --prob_statio 0.9 --cl_strategy loss --cl_tbd_thres -1 --num_epochs 1 --num_ways 10 --num_shots=3 --n_runs 1
#python main.py -v --prob_statio 0.9 --num_epochs 1 --num_ways 10 --num_shots=3 --num_epochs 10 --n_runs 10
