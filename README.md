# SOVA NLU

SOVA NLU is an intent classification solution based on [BERT](https://arxiv.org/abs/1810.04805) architecture. It is designed as a REST API service and it can be customized (both code and models) for your needs.

## Installation

The easiest way to deploy the service is via docker-compose, so you have to install Docker and docker-compose first. Here's a brief instruction for Ubuntu:

#### Docker installation

*	Install Docker:
```bash
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $(whoami)
```
In order to run docker commands without sudo you might need to relogin.
*   Install docker-compose:
```
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

*   (Optional) If you're planning on using CUDA run these commands:
```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```
Add the following content to the file **/etc/docker/daemon.json**:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
Restart the service:
```bash
sudo systemctl restart docker.service
``` 

#### Build and deploy

*   Checkout repository and models from Git LFS (you can also download pre-trained models [manually](https://disk.yandex.ru/d/Gs2jYisawiAj6w)):
    ```bash
    git clone https://github.com/sovaai/sova-nlu
    cd sova-nlu
    git lfs install
    git lfs pull
    ```

*   Build docker image
     *   If you're planning on using CPU only: build *sova-nlu* image using the following command:
     ```bash
     sudo docker-compose build
     ```
     *   If you're planning on using GPU: modify `Dockerfile`, `docker-compose.yml` (uncomment the runtime and environment sections) and build *sova-nlu* image:
     ```bash
     sudo docker-compose build
     ```

*   Run web service in a docker container
     ```bash
     sudo docker-compose up -d sova-nlu
     ```

## Testing

To test the service you can send a POST request:
```bash
curl --request POST 'http://localhost:8000/get_intent' --header "Content-Type: application/json" --data '{"text": "Включи режиссерскую версию Лиги справедливости"}'
```

You can also use web interface by opening http://localhost:8000/docs.

## Training

*   Use the same Docker image that was already built for the service. Customize the hyperparameters in `config.py` and use your own `data/data.csv` for training.
*   To start training simply run:
     ```bash
     sudo docker-compose up -d sova-nlu-train
     ```
The trained model will be saved to `data/intent_classifier.pt`.

## Licenses
SOVA NLU is licensed under Apache License 2.0 by Virtual Assistant, LLC.
