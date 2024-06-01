# codeGen
Repo for a general framework to test multipl;e conde generation LLMs.

## Supported Models

* Codestral

## Quick Start From Docker Image
* Pull docker image from registry 
```bash
docker pull docker pull huggingface/transformers-pytorch-gpu

docker run -it --rm -p 9999:8888 -v $(pwd):/opt/app -v [path to data]:/opt/app/data --shm-size=20G docker pull huggingface/transformers-pytorch-gpu
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

