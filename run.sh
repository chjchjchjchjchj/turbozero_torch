python turbozero.py --verbose --gpu --mode=train --config=./example_configs/othello_mini.yaml --logfile=./othello_mini.log
CUDA_VISIBLE_DEVICES=1 python turbozero.py --verbose --gpu --mode=train --config=./example_configs/vector_selection.yaml --logfile=./vector_selection.log
python turbozero.py --verbose --gpu --mode=train --config=./example_configs/vector_selection_cos.yaml --logfile=./vector_selection_cos.log


torchrun --nproc_per_node=8 turbozero_multi.py \
    --verbose \
    --gpu \
    --mode=train \
    --config=./example_configs/vector_selection.yaml \
    --logfile=./vector_selection.log


CUDA_VISIBLE_DEVICES=1 python turbozero.py --verbose --gpu --mode=train --config=./example_configs/vector_selection.yaml --logfile=./vector_selection_mcts_iter_100.log
