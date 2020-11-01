# MAML
# Algo4, always task shift and ood, so overfit slow-weight to current task, no UM
#python main.py -v --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1
# No UM
#python main.py -v --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss'
# With UM
#python main.py -v --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0
# Algo3, with UM
#python main.py -v --prob_statio 0.9 --num_ways 5 --num_shots=3 --num_epochs 1 --n_runs 1 --cl_tbd_thres 1.0 --cl_strategy_thres 3.0 --cl_strategy 'loss' --um_power 1.0 --algo3 True



# My (brandon) wandb login command:
# wandb login ab6b3d140b9830ddd9b362451e48787f66410180

# Proto-MAML
#python main.py -v \
#  --wandb "cs330_finalproject" \
#  --name "protomaml" \
#  --prob_statio 0.9 \
#  --cl_strategy loss \
#  --cl_tbd_thres 0.5 \
#  --num_epochs 1 \
#  --num_shots=3 \
#  --model_name protomaml \
#  --use_different_nway True
# MAML
python main.py -v \
  --wandb "cs330_finalproject" \
  --name "maml" \
  --prob_statio 0.9 \
  --cl_strategy loss \
  --cl_tbd_thres 0.5 \
  --num_epochs 1 \
  --num_shots=3
