export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

log_name=log_bert
nohup python -u  driver/Train.py --config_file expdata/opinion.cfg --thread 1 > $log_name 2>&1 &
tail -f $log_name