#!/bin/bash

############################################
##            ENV & JOB CONFIG            ##
############################################

config=./scripts/config.json
proj_dir=$(jq -r '.DIR.PROJ_DIR' $config)
# check model and dataset
model=$(jq -r '.ML_CONFIG.MODEL2' $config)
dataset=$(jq -r '.ML_CONFIG.DATA' $config) # mnist | fmnist | cifar10 | cifar100 | shakespeare
iid=$(jq -r '.FL_CONFIG.IID' $config) # all dataset above only have noniid case except cifar10

data_home=$(jq -r '.DIR.DATA_DIR' $config)"/$dataset"

# ip of server
this_addr=$(jq -r '.ADDR.HOSTIP') # the host ip

# imagename
imagename=$(jq -r '.ETC.IMAGENAME' $config)

#for validation
conda=$(jq -r '.ETC.CONDA_DIR' $config)
conda_env=$(jq -r '.ETC.CONDA_ENV' $config)
############################################

s_flag=0
i_flag=0
n_flag=0
abs_i=0
# get argument
opti=$#
array=("$@")

gpu_nums=$(nvidia-smi -L | grep -c UUID)

# function for make, run, erase container
function RUN() {
    echo "create $N container"
    i=0
    PID=27000
    while [ "$i" -lt "$n" ]; do
        CID=$((i+abs_i))
        echo $CID
        GID=$((i%gpu_nums))
        RID=$(($PID+49))
        docker run -it --cap-add=NET_ADMIN --name "ct$CID" -d -m 12000m --oom-kill-disable -p 0.0.0.0:$PID-$RID:$PID-$RID --cpuset-cpus="$i" --gpus "device=$GID" --cap-add ALL --ipc=host $imagename

        if [ "$dataset" != "shakespeare" ]; then
        docker cp $data_home/$iid/$CID "ct$CID":/workspace/data/dat/train/$CID
        else
        # for shakespeare dataset
        docker cp $data_home/train_$iid/train_$CID.pkl "ct$CID":/workspace/data/dat/train/
        docker cp $data_home/test_$iid/test_$CID.pkl "ct$CID":/workspace/data/dat/test/
        fi

        i=$(( i+1 ))
        PID=$(( PID+50 ))
    done
    rm -rf ~/fed_git
}
function EXEC() {
    helper_addr="${array[7]}"
    helper_ip=$(echo "$helper_addr"| grep -oE "[[:digit:]]{1,}\.[[:digit:]]{1,}\.[[:digit:]]{1,}\.[[:digit:]]{1,}" )
    helper_port=$(echo "$helper_addr"| grep -oE "[[:digit:]]{1,}$" )
    if [ $s_flag -eq 0 ]; then
    echo "Error: Options -S MUST be required."
    usage
    fi
    v=$(echo "$helper_addr" | grep -oE "[[:digit:]]{1,}\.[[:digit:]]{1,}\.[[:digit:]]{1,}\.[[:digit:]]{1,}:[[:digit:]]{1,}$" )
    echo "$v"

    if [ "$v" != "$helper_addr" ]; then
    echo "Error: Helper addr must be required."
    usage
    fi

    echo "execute $n container"
    i=0
    while [ "$i" -lt "$n" ]; do
        CID=$((i+abs_i))
        PID=$(( helper_port+i*50 ))
        if [ $((i+abs_i)) -eq 0 ]; then
            echo '0 start'
            CMD="python -u P2P-Fed.py -p $helper_port -t $CID -s $s -g 0 -a $this_addr -i -m $model -d $dataset -c --slow ${array[8]} ${array[9]}"
        else
            CMD="python -u P2P-Fed.py -p $PID -P $helper_port -t $CID -s $s -g 0 -A $helper_ip -a $this_addr -i -m $model -d $dataset -c --slow ${array[8]} ${array[9]}"
        fi
        echo "$CMD"
        PORT=$((PID+49))
        log=$proj_dir/log/DFL_$CID.log
        date > $log
        docker exec "ct$CID" $CMD >> $log 2>&1 &  
        i=$(( i+1 ))
    done
}
function REMOVE() {
    pids=()
    echo "Erase container"
    i=0
    while [ "$i" -lt "$n" ]; do
        CID=$((i+abs_i))
        echo "erase ct$CID"
        docker rm -f "ct$CID"
        pids+=($!)
        i=$(( i+1 ))
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "Delete saved models"
    rm -r $proj_dir/model/mdls/*

    echo "All operations completed. Exiting."
}
function GET() {
    pids=()
    echo "Stop container and Get nets"
    i=0
    while [ "$i" -lt "$n" ]; do
        CID=$((i+abs_i))
        echo "Stop ct$CID"
        docker stop "ct$CID"&
        pids+=($!)
        i=$(( i+1 ))
    done

    i=0
    while [ "$i" -lt "$n" ]; do
        CID=$((i+abs_i))
        docker cp "ct$CID":/workspace/mdls/ $proj_dir/model/ &
        pids+=($!)
        i=$(( i+1 ))
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    source $conda $conda_env
    cd $proj_dir
    pids=()
    for ((k=0; k < gpu_nums; k++)) do
        python load.py -n "$s" -g "$gpu_nums" -u "$k" -m 0 -d "$dataset"&
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    cd ..

    echo "$(hostname) operations completed. Exiting."
}

# functions just for shell
function usage() {
  echo ""
  echo "Usage: $0 [options] address_of_helper_node:port"
  echo -e "\nOptions: \n  -S\tNumber of all client\n  -N\tNumber of container to build"
  echo -e "  -I\tDocker's Index number\n  -C\tRun containers"
  echo -e "  -E\tExcute the job\n  -R\tRemove container"
  echo ""
  exit
}
function error_check() {
    if [[ $OPTARG != *[0-9]* ]];then
    echo "argtest: failed to parse argument: '$OPTARG': Invalid argument"
    exit
    fi
}
function optarg_check() {
    if [ $n_flag -eq 0 ] || [ $i_flag -eq 0 ]; then
    echo "Error: Options -S, -N, and -I MUST be required."
    usage
    fi
}

if [[ $opti -eq 0 ]]; then
    usage
fi

# parse option
while getopts :S:N:I:CE:RGh opt
do
    case $opt in
        h)
            usage
        ;;
        S) 
            error_check
            s=$OPTARG
            s_flag=1
        ;;
        N) 
            error_check
            n=$OPTARG
            n_flag=1
        ;;
        I) 
            error_check
            abs_i=$OPTARG
            i_flag=1
        ;;
        C) 
            optarg_check
            RUN
        ;;
        E) 
            optarg_check
            EXEC
        ;;
        R) 
            optarg_check
            REMOVE
        ;;
	    G)
	    optarg_check
	    GET
        ;;
        *) echo "$opt is not the option"
        usage;;
        esac
done



