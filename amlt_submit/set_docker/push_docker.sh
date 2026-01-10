az acr login --name workspacegenaiacr.azurecr.io
docker tag nvcr.io/nvidia/pytorch:25.11-py3-amlt  workspacegenaiacr.azurecr.io/pytorch:25.11-py3-amlt 
docker push workspacegenaiacr.azurecr.io/pytorch:25.11-py3-amlt