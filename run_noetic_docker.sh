docker run -it \
    --gpus all \
    --privileged \
    --network host \
    --env="DISPLAY" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/home/yunlu/work:/home/work:rw" \
    wbmp:latest \
    /bin/bash