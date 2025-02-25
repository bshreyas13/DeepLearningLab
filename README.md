# Deep Learning Lab
The project contains a framework for easily testing opensource-deep learning models from hugging face.
Currenlty only supports inference ing pretrained weights or incontext-learning. Finetuning will be added eventually.  
You will need to clone the HF repo for any model of interest currently. 

## Supported Models

## Supported Models
For supported models please refer to the [supported_models.json](./llmLab/supported_models.json) file.

## Quick Start From Docker Image

* Step 1: Clone the repository to local machine or cluster compute node.
```bash
git clone git@github.com:bshreyas13/DeepLearningLab.git
```

* Step 2: cd to downloaded repository.
```bash
cd DeepLearningLab
```

* Step3a: Pull docker image from registry 
```bash
docker pull bshreyas13/huggingface_docker:latest
```
```bash
docker run -e HF_TOKEN=XXXXX -it --rm -p 9999:8888 --user=12764:10001 -v $(pwd):/opt/app -v /vtti:/vtti --gpus all --shm-size=60G hfdocker:latest
```
## NOTE: Generating Hugging Face Token
To generate a Hugging Face token(`HF_TOKEN`), follow these steps:

1. Go to the Hugging Face website (https://huggingface.co/).
2. Sign in to your account or create a new one if you don't have an account.
3. Once you are signed in, click on your profile picture in the top right corner and select "setting" from the dropdown menu. Then "Access Token" on the left of the screen.
4. On the Token page, click on the "New token" button.
5. Enter a name for your token (e.g., "My Token") and click on the "Create" button.
6. Your token will be generated and displayed on the Token page. Make sure to copy and securely store your token as it will be required for authentication when using Hugging Face APIs or services.

That's it! You have successfully generated a Hugging Face token. Remember to keep your token confidential and avoid sharing it with others.


* Step 3b: If yoou already pulled the image ignore the following steps to build the docker image using Dockerfile. 
```bash
docker build -e HF_TOKEN=XXX -f Dockerfile -t hfcodegen .
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
Once in the docker continer terminal use the `start_chat.py` script to start chatting with model specified by `model_path`
Example:
```bash
python single_model_demo.py --model_path "/vtti/projects06/451857/Data/Dump/ShreyasTest/Meta-Llama-3-8B-Instruct" --quantize --log_path "./LOGS"
```
**NOTE** : currently requires `--quantize` flag, Large model inference optimizations in progress.


## Authors and acknowledgment

**Author**:

* Name: Shreyas Bhat
* Email: sbhat@vtti.vt.edu

* Name: Sarang Joshi
* Email: sarang87@vt.edu

* Name : 
* Email :


**Maintainers**

* Name: Sarang Joshi
* Email: sarang87@vt.edu

* Name : 
* Email :

* Name: Shreyas Bhat
* Email: sbhat@vtti.vt.edu


## License
For open source projects, say how it is licensed.

## Project status And roadmap
- Add base inference wrapper    
    1. Load the tokenizer for the model. -Done
    2. Load the model from the local model path. -Done
    3. Infer the device map for the model. -Done
    4. Dispatch the model to the specified devices in the device map. -Done
    5. Generate text based on the given chat. -Done
    6. Test with deepseek : https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct -Done
    7. Find a to use the quamtized coderstral/codellamma models - NA
    8. Post issue for code-gemma -Done
    9. Test with code-gemma : https://huggingface.co/google/codegemma-1.1-7b-it -Done
    10. codestral hf space: https://huggingface.co/spaces/poscye/code - Pending
    11. check why not distributing across GPUS - Pending
    12. Since ditribution fails unable to infer with out `--quantize` - Pending
    13. Paligemma langchain integration - Done
    14. Test paligemma in chat more rigourously - In progress
    15. Test model stacking ideas - In progress
    16. Added LLama3 for chat capability - Done
- Add support for object detectors - test with DETR, pending
- Add support for training/finetuning
- Add support for API calls openAI, gemini - In progress 
- Use langchain to build chat bot - done
- Build/Add similar framework for Vision language models - Itegrated and basic testing complete.
- Automate repo cloning - Pending
- Basic 1 document RAG project - Future work
- Generalize a spohisticated hsitory management system

# Citation

1. Meta AI Llama3
```
@article{llama3modelcard,

title={Llama 3 Model Card},

author={AI@Meta},

year={2024},

url = {https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md}

}

```
2. Google Paligemma 
```
@article{paligemmamodelcard,

title={Paligemma-3b-pt-334},

author={Google},

year={2024},

url = {https://huggingface.co/google/paligemma-3b-pt-224}

}
```
