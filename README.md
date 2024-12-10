# Efficient Streaming Language Models with Attention Sinks with Retrieval Augmented Generation

Fork of https://github.com/mit-han-lab/streaming-llm for MIT 6.5940 final project.

## Demo

[Watch the demo on Google Drive](https://drive.google.com/file/d/1dGzbcO3nt8dae7uERvL4-8KWxn-LmqlU/view)

## Usage

### Environment Setup

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

conda install -c conda-forge faiss-gpu

python setup.py develop
```

### Run Streaming Llama Chatbot

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py  --enable_streaming
```

### Run Streaming Llama Chatbot with RAG

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py  --enable_streaming --enable_rag
```
