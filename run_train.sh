./clear.sh
#python3 rm_weights_i_logs.py 45
export CUDA_VISIBLE_DEVICES=0
python3.5 main_torch_latest.py 0
