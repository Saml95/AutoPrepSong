docker build -t nvcr.io/nvidia/pytorch:25.11-py3-amlt-AutoPrepSongV2-diarize .

az acr login --name workspacegenaiacr.azurecr.io
docker tag nvcr.io/nvidia/pytorch:25.11-py3-amlt-AutoPrepSongV2-diarize  workspacegenaiacr.azurecr.io/pytorch:25.11-py3-amlt-AutoPrepSongV2-diarize
docker push workspacegenaiacr.azurecr.io/pytorch:25.11-py3-amlt-AutoPrepSongV2-diarize