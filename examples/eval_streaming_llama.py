import warnings

warnings.filterwarnings("ignore")

import torch
# import numpy as np
import argparse
import json
import os
import time
import re
import sys
import faiss

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl, embed_text
from streaming_llm.enable_streaming_llm import enable_streaming_llm

from run_streaming_llama import retrieve_from_db


@torch.no_grad()
def calculate_perplexity_with_cache(model, tokenizer, input_ids, past_key_values, response_ids, kv_cache=None):
    """
    Calculate the perplexity of the response conditioned on the prompt using a cache.

    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        prompt: The input prompt (string).
        response: The reference response (string).
        kv_cache: Key-value cache for attention (optional).

    Returns:
        perplexity: The perplexity of the response given the prompt.
        kv_cache: Updated key-value cache after processing the sequence.
    """
    # Tokenize the prompt and response together
    # prompt_text = f"USER: {prompt}\n\nASSISTANT: "
    # input_text = prompt + response
    # print(input_text)
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    # Tokenize only the response for reference token indices
    # response_ids = tokenizer(response, return_tensors="pt").input_ids.to(model.device)

    # Feed the input to the model with the cache
    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values
    # print('test')
    # Extract log probabilities for response tokens
    response_start_idx = input_ids.shape[1] - response_ids.shape[1] - 1   # Start of the response in logits
    log_probs = torch.nn.functional.log_softmax(logits[:, response_start_idx - 1:-1, :], dim=-1)
    target_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

    # Calculate average log probability and perplexity
    avg_log_prob = target_log_probs.mean()
    perplexity = torch.exp(-avg_log_prob).item()

    return perplexity, past_key_values

@torch.no_grad()
def streaming_inference_rag_with_perplexity(model, tokenizer, prompts, responses, index, kv_cache=None, max_gen_len=1000):
    """
    Perform inference and calculate perplexity for responses conditioned on prompts.

    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        prompts: List of prompts (strings).
        responses: List of responses (strings).
        kv_cache: Key-value cache for attention (optional).
        max_gen_len: Maximum generation length.

    Returns:
        perplexities: List of perplexities for each prompt-response pair.
    """
    past_key_values = None
    perplexities = []
    history_token_ids = []
    for idx, prompt in enumerate(prompts):
        prompt_text = ''
        retrieved_docs = retrieve_from_db(model, tokenizer, index, prompt)
        if retrieved_docs:
            prompt_text += (
                "\nProviding below some context that you may or may not find useful. \n"
            )
            prompt_text += "CONTEXT: "
            for i, doc in enumerate(retrieved_docs):
                prompt_text += "\n<context {}> {} </context {}>".format(
                    i + 1, doc.strip("\n"), i + 1
                )
            prompt_text += '\n'
        # formatted_prompt += "\nASSISTANT: "


        prompt_text += "USER: " + prompt + "\nASSISTANT: " + responses[idx] + '\n'
        # print("\n" + prompt_text, end="")
        
        response_ids = tokenizer(responses[idx], return_tensors="pt").input_ids.to(model.device)

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        history_token_ids += input_ids.tolist()[0]
        input_ids = input_ids.to(model.device)
        # print(input_ids)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len
            past_key_values, history_token_ids = kv_cache.evict_for_space(
                tokenizer,
                past_key_values,
                space_needed,
                history_token_ids
            )
        # Calculate perplexity for the response
        # response = responses[idx]
        perplexity, past_key_values = calculate_perplexity_with_cache(
            model, tokenizer, input_ids, past_key_values, response_ids, kv_cache
        )

        # history_token_ids += generated_ids
        # add prompt to index
        embedded_chunk = embed_text(prompt, model, tokenizer)
        index.add(embedded_chunk)
        with open("data/embeddings.txt", "a") as file:
            file.writelines([prompt.strip("\n") + '\n'])

        perplexities.append(perplexity)
        # print(f"\nPerplexity: {perplexity}")
    return perplexities


@torch.no_grad()
def streaming_inference_with_perplexity(model, tokenizer, prompts, responses, kv_cache=None, max_gen_len=1000):
    """
    Perform inference and calculate perplexity for responses conditioned on prompts.

    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        prompts: List of prompts (strings).
        responses: List of responses (strings).
        kv_cache: Key-value cache for attention (optional).
        max_gen_len: Maximum generation length.

    Returns:
        perplexities: List of perplexities for each prompt-response pair.
    """
    past_key_values = None
    perplexities = []
    history_token_ids = []
    for idx, prompt in enumerate(prompts):
        prompt_text = "USER: " + prompt + "\nASSISTANT: " + responses[idx] + '\n'
        
        # print(prompt_text)
        
        response_ids = tokenizer(responses[idx], return_tensors="pt").input_ids.to(model.device)

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        history_token_ids += input_ids.tolist()[0]
        input_ids = input_ids.to(model.device)
        # print(input_ids)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len
            past_key_values, history_token_ids = kv_cache.evict_for_space(
                tokenizer,
                past_key_values,
                space_needed,
                history_token_ids
            )

        # Calculate perplexity for the response
        # response = responses[idx]
        perplexity, past_key_values = calculate_perplexity_with_cache(
            model, tokenizer, input_ids, past_key_values, response_ids, kv_cache
        )
        perplexities.append(perplexity)
        # print(f"\nPerplexity: {perplexity}")
    return perplexities



def main(args):
    model_name_or_path = args.model_name_or_path
    dataset_name = args.dataset_name
    max_gen_len = args.max_gen_len
    model, tokenizer = load(model_name_or_path)
    test_filepath = os.path.join(args.data_root, "{}.jsonl".format(dataset_name))
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    responses = []
    for sample in list_data:
        text = sample['text']
        # text = data.get("text", "")
        prompt_start = text.find("<human>:") + len("<human>:")
        response_start = text.find("<bot>:")
        
        prompt = text[prompt_start:response_start].strip()
        response = text[response_start + len("<bot>:"):].strip()
        prompts.append(prompt)
        responses.append(response)
    # print(prompts)
    # print(responses)
    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    with open("data/evicted.txt", "w") as file:
        pass
    if args.enable_rag:
        embeddings_dimension = None
        if "Llama-2-7b" in args.model_name_or_path:
            embeddings_dimension = 4096
        elif "Llama-2-13b" in args.model_name_or_path:
            embeddings_dimension = 5120
        elif "vicuna-7b-v1.3" in args.model_name_or_path:
            embeddings_dimension = 4096
        else:
            raise "Unknown embeddings_dimension"

        index = faiss.IndexFlatL2(embeddings_dimension)
        with open("data/embeddings.txt", "w") as file:
            pass

        if index.ntotal == 0:
            print("Vector DB is initialized and is empty.")

        perplexities = streaming_inference_rag_with_perplexity(
            model,
            tokenizer,
            prompts,
            responses,
            index,
            kv_cache,
            max_gen_len,
        )
    else:
        perplexities = streaming_inference_with_perplexity(
            model,
            tokenizer,
            prompts,
            responses,
            kv_cache,
            max_gen_len=max_gen_len
        )
    print(f"\nPerplexity Mean: {sum(perplexities) / len(perplexities)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/llama-2-7b-chat-hf"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--dataset_name", type=str, default="unified_chip2_small")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--enable_rag", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    parser.add_argument("--max_gen_len", type=int, default=1000)
    args = parser.parse_args()

    main(args)
