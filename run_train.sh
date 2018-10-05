#./clear.sh # If you want to train from scratch again, uncomment it
#python3 rm_weights_i_logs.py 45 # if you want to re-train from specific epoch, uncomment it and change the number to your starting epoch, but you need to make sure that the corresponding weights are in the save_weights/ directory.
export CUDA_VISIBLE_DEVICES=0 # Change the GPU you want to use.
python3.5 main_torch_latest.py 0
