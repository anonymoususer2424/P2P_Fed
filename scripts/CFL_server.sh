#!/bin/bash
opti=$#
array=("$@")

config=./scripts/config.json
proj_dir=$(jq -r '.DIR.PROJ_DIR' $config)
# imagename
imagename=$(jq -r '.ETC.IMAGENAME' $config)
if [[ $opti -lt 2 ]]; then
    echo "\$CMD \$log must be given!"
    exit
fi

CMD="${array[@]:0:$opti-1}"
log="${array[$opti-1]}"
echo "$CMD"

docker run -it --cap-add=NET_ADMIN --name "server" -d -p 0.0.0.0:16000-16049:16000-16049 -m 12000m --oom-kill-disable --cpuset-cpus="17" \
 --gpus "device=0" --cap-add ALL --ipc=host $imagename

docker cp "$proj_dir" "server":/workspace
date > "$log"
docker exec "server" $CMD >> $log 2>&1 &
