# codeGen
Repo for a general framework to test multipl;e conde generation LLMs.

## Supported Models

* Codestral

## Quick Start From Docker Image

* Pull docker image from registry 
```bash
docker pull docker pull <Updated link>
```
```bash
docker run -e HF_TOKEN=XXXXX -it --rm -p 9999:8888 --user=12764:10001 -v $(pwd):/opt/app -v /vtti:/vtti --gpus all --shm-size=60G hfdocker:latest
```

* Step 1: Clone the repository to local machine or cluster compute node.
```bash
git clone git@github.com:bshreyas13/codeGen.git
```
* Step 2: cd to downloaded repository.
```bash
cd codeGen
```
* Step 3: Build the docker image using Dockerfile. **Note**: use the Dockerfile in the home directory of the repo since it has more updated instruction than the one in mmdetection/docker.
```bash
docker build -f Dockerfile -t hfcodegen .
```
* Step 4: Run container from image and mount data volumes.
```bash
docker run -e HF_TOKEN= 'you hugging face acccess token' -it --rm -p 9999:8888 -v $(pwd):/opt/app -v [path to data]:/opt/app/data --shm-size=20G hfcodegen
```
For example: 
```bash
docker run -e HF_TOKEN= 'you hugging face acccess token' -it --rm -p 9999:8888 --user=12764:10001 -v $(pwd):/opt/app -v /vtti:/vtti --gpus all --shm-size=20G hfcodegen
```
* To run in dettached mode use the `-d` flag. This will ensure the conatiner continues to run irrespective of terminal status
For example: 
```bash
docker run -d -e HF_TOKEN=XXXXX -it --rm -p 9999:8888 --user=12764:10001 -v $(pwd):/opt/app -v /vtti:/vtti --gpus all --shm-size=60G hfdocker:latest hfcodegen
docker ps
docker attach <name of your docker container from ps command>
```

**NOTE**: You may get an error `failed: port is already allocated`. If so, expose a different port number on the server, e.g. '9898:8888'
* If you wish to run the jupyter notebook, type 'jupyter' on the container's terminal
* On your local machine perform port forwarding using
```bash
ssh -N -f -L 9999:localhost:9999 host@server.xyz
```

## Usage

Example:
```bash
python modelManager.py --quantize
```
## Roadmap
- Add base inference wrapper    
    1. Load the tokenizer for the model.
    2. Load the model from the local model path.
    3. Infer the device map for the model.
    4. Dispatch the model to the specified devices in the device map.
    5. Generate text based on the given chat.
    6. Test with deepseek : https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct
    7. Find a to use the quamtized coderstral/codellamma models
    8. Post issue for code-gemma
    9. Test with code-gemma : https://huggingface.co/google/codegemma-1.1-7b-it
- Add support for training/finetuning
- Add support for models openAI, gemini eventually. 
- Use langchain to build chat bot
- Build similar framework for Vision language models

## Authors and acknowledgment

**Author**:
* Name: Sarang Joshi
* Email: sarang87@vt.edu

* Name : 
* Email :

* Name: Shreyas Bhat
* Email: sbhat@vtti.vt.edu

**Maintainers**

* Name: Sarang Joshi
* Email: sarang87@vt.edu

* Name : 
* Email :

* Name: Shreyas Bhat
* Email: sbhat@vtti.vt.edu


## License
For open source projects, say how it is licensed.

## Project status
* Docker built 


# Citation

