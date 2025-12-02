export CUBLAS_WORKSPACE_CONFIG=:4096:8

local_path='<your experiment path>'


python3 -u main.py --config=./configs/moe_openset/cf100.yaml \
	--seed 1993 \
	--local_path=$local_path
