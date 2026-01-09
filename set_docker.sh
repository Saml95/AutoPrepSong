

# Evaluation
# name="AutoPrep"
name="AutoPrepSongV2"
# container="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
# name=tmp
sudo docker stop $name
sudo docker rm $name
# container="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
# container="nvcr.io/nvidia/pytorch:24.07-py3"
# container="nvcr.io/nvidia/pytorch:24.10-py3"
container="nvcr.io/nvidia/pytorch:25.11-py3"

sudo docker run -d --privileged  --name $name \
    --net=host  --ipc=host  --shm-size 900g --shm-size 300g --gpus all \
    --user root \
    -v /home/:/home/ \
    -v /tmp:/tmp \
    -v /data/:/data/ \
    $container /bin/bash -c "sleep infinity "





sudo docker exec -it $name bash -c "bash"
