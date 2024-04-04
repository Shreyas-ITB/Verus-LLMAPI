# Verus-LLMAPI
A Large Language Model API for the community chatbots at Verus

## Requirements

- 13B model requires a VM with atleast 16GB of RAM and a GPU with 8GB VRAM. A Linux VM is appreciated.
- 7B model requires a VM with atleast 8GB of RAM and a GPU with 6GB VRAM, again a Linux VM is appreciated.
- Python 3.10.11 installed with latest version of pip.
- Git is required for obvious reasons.
- Model files from huggingface (gguf only), supported models can be seen [here](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#description). I highly recommend using [LlaVa 1.6 7B](https://huggingface.co/cjpais/llava-v1.6-vicuna-7b-gguf/blob/main/llava-v1.6-vicuna-7b.Q5_K_M.gguf) or the [LlaVa 1.6 13B](https://huggingface.co/cjpais/llava-v1.6-vicuna-13b-gguf/blob/main/llava-v1.6-vicuna-13b.Q5_K_M.gguf)

## Setup

- Rename the `example.env` to `.env` and edit the file accordingly.
```
PERSIST_DIRECTORY=db # directory of the datasets/vectorstores.
MODEL_PATH=models/llava-v1.6-vicuna-13b.Q5_K_M.gguf # Place the models in the models directory and name the model you are using there.
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2 # embeddings model name (will be fetched automatically if found) leave it as it is.
MODEL_N_CTX=1000 # Maximum token limit for the LLM model
MODEL_N_BATCH=1024 # Number of tokens in the prompt that are fed into the model at a time. Optimal value differs a lot depending on the model (1024 is better)
LLMAPI_HOST=127.0.0.1 # API Host URL (set it to 0.0.0.0 if you are running it in production)
LLMAPI_PORT=5000 # Port for the API, leave it as it is.
TARGET_SOURCE_CHUNKS=4 # The amount of chunks (sources) that will be used to answer a question
WHISPER_MODEL=base # Whisper model that the code is going to use to transcribe audio files to create vector scores.
```

- Install the dependencies
```pip install -r requirements.txt```

- You would also need `ffmpeg` installed on your linux system or if you are trying this on windows then download the latest version of `ffmpeg.exe` on the root directory of this project.

- Download the [datasets](https://github.com/Shreyas-ITB/VerusDatasets) and put the files under source_documents.

- Run the LLMAPI
```python3 llmapi.py```
Now the model should automatically create vectors and run the API for you.

## Donations
- VRSC: [Shreyas-ITB@](https://insight.verus.io/address/Shreyas-ITB@)
