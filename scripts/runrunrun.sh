#!/bin/bash

#####################################
# PARAMETER
config=./scripts/config.json
proj_dir=$(jq -r '.DIR.PROJ_DIR' $config)

slow=$(jq -r '.FL_CONFIG.SLOW' $config) # number of staggler

model=$(jq -r '.ML_CONFIG.MODEL' $config)
data=$(jq -r '.ML_CONFIG.DATA' $config)

if [ "$data" = "cifar100" ];then
outsize=100
else
outsize=10
fi

runtime=$(jq -r '.FL_CONFIG.RUNTIME' $config)
using_host=($(jq -r '.ADDR.NODENUM[]' $config))
node=0
for nn in "${using_host[@]}"
do
node=$(( node+nn )) 
done

conda=$(jq -r '.ETC.CONDA_DIR' $config)
conda_env=$(jq -r '.ETC.CONDA_ENV' $config)
source $conda $conda_env
#####################################

bash scripts/run_DFL.sh -C
sleep 10
bash scripts/run_DFL.sh -E $slow
sleep $((runtime+100)) # because of starting time
bash scripts/run_DFL.sh -G
sleep 50
bash scripts/run_DFL.sh -R
sleep 15

CMD="python -u FedAvg_server.py -c $node -A 0 -r 10 -o $outsize -M $model -s $slow"
log="$proj_dir/log/fedavg.log"
echo $log
bash scripts/CFL_server.sh $CMD $log

bash scripts/run_CFL_client.sh -C
sleep 10
bash scripts/run_CFL_client.sh -E
sleep $runtime
bash scripts/run_CFL_client.sh -R
sleep 50

docker stop server
docker cp server:/workspace/mdls $proj_dir/model/

python $proj_dir/load.py -u 0 -g 1 -n $node -m 1 -d $data
docker rm server -f

sleep 15

CMD="python -u FedAvg_server.py -c $node -A 1 -r 10 -o $outsize -M $model -s $slow"
log="$proj_dir/log/fedbuff.log"
bash scripts/CFL_server.sh $CMD $log

bash scripts/run_CFL_client.sh -C
sleep 100
bash scripts/run_CFL_client.sh -E
sleep $runtime
bash scripts/run_CFL_client.sh -R
sleep 50

docker stop server
docker cp server:/workspace/mdls $proj_dir/model/

python $proj_dir/load.py -u 0 -g 1 -n $node -m 1 -d $data
docker rm server -f
rm -r  $proj_dir/model/*
sleep 15

CMD="python -u FedAvg_server.py -c $node -A 2 -r 10 -o $outsize -M $model -s $slow"
log="$proj_dir/log/fedasync.log"
bash scripts/CFL_server.sh $CMD $log

bash scripts/run_CFL_client.sh -C
sleep 100
bash scripts/run_CFL_client.sh -E
sleep $runtime
bash scripts/run_CFL_client.sh -R
sleep 50

docker stop server
docker cp server:/workspace/mdls $proj_dir/model/

python $proj_dir/load.py -u 0 -g 1 -n $node -m 1 -d $data
docker rm server -f
rm -r  $proj_dir/model/*
sleep 15

CMD="python -u FedAvg_server.py -c $node -A 3 -r 10 -o $outsize -M $model -s $slow"
log="$proj_dir/log/fedprox.log"
bash scripts/CFL_server.sh $CMD $log

bash scripts/run_CFL_client.sh -C
sleep 100
bash scripts/run_CFL_client.sh -E
sleep $runtime
bash scripts/run_CFL_client.sh -R
sleep 50

docker stop server
docker cp server:/workspace/mdls $proj_dir/model/

python $proj_dir/load.py -u 0 -g 1 -n $node -m 1 -d $data
docker rm server -f
rm -r  $proj_dir/model/*
sleep 15

bash scripts/run_DFL.sh -C
sleep 100
bash scripts/run_DFL.sh -E $slow --dfedavgm
sleep $((runtime+100)) # because of starting time
bash scripts/run_DFL.sh -G
sleep 100
bash scripts/run_DFL.sh -R

