#--prob_statio
#--cl_strategy 'loss'
#--cl_strategy_thres 0.1  # determine ood
#--cl_tbd_thres -1 (task boundary, loss change)

# Proto-MAML
python main.py -v --prob_statio 0.9 --cl_strategy loss --cl_tbd_thres 0.5 --num_epochs 1 --num_shots=3 --model_name protomaml --use_different_nway True
# MAML
#python main.py -v --prob_statio 0.9 --cl_strategy loss --cl_tbd_thres 0.5 --num_epochs 1 --num_shots=3
