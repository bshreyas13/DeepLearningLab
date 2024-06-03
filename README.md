# codeGen
Repo for a general framework to test multipl;e conde generation LLMs.

## Supported Models

* Codestral

## Quick Start From Docker Image

* Pull docker image from registry 
```bash
docker pull docker pull huggingface/transformers-pytorch-gpu
```
```bash
docker run -e HF_TOKEN=XXXXX -it --rm -p 9999:8888 --user=12764:10001 -v $(pwd):/opt/app -v /vtti:/vtti --gpus all --shm-size=60G hfdocker:latest
```

## Usage

## Roadmap
- Add base inference wrapper
- Add support for training/finetuning
- Add support for models openAI, gemini eventually.

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

