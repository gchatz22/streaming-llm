import torch
from streaming_llm.utils import embed_text
import re

def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, tokenizer, past_key_values, num_coming, history_token_ids):
        if past_key_values is None:
            return None, history_token_ids
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values, history_token_ids
        
        evicted_tokens_index_start = self.start_size
        evicted_tokens_index_end = seq_len - self.recent_size + num_coming
        evicted_tokens_ids = history_token_ids[
            evicted_tokens_index_start:evicted_tokens_index_end
        ]
        evicted_tokens_text = " ".join(
            tokenizer.decode(
                evicted_tokens_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )
        
        with open("data/evicted.txt", "a") as file:
            file.writelines([evicted_tokens_text.strip("\n")])
        
        history_token_ids = (
            history_token_ids[0:evicted_tokens_index_start]
            + history_token_ids[evicted_tokens_index_end:]
        )
        
        new_past_key_values = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
        return new_past_key_values, history_token_ids

    def evict_for_space_rag(
        self, model, tokenizer, index, past_key_values, num_coming, history_token_ids
    ):
        if past_key_values is None:
            return None, history_token_ids
        # print(past_key_values[0][0].shape) # (torch.Size([1, 32, 46, 128]))
        # (batch_size, num_heads, sequence_length, embed_size_per_head)
        # Llama2 7B has 32 attention heads34. Each of its 32 layers contains 32 attention heads, resulting in a total of 1024 attention head components7
        # For Llama2 7B, the embedding size per attention head is 128. This can be calculated by dividing the total hidden size (4096) by the number of attention heads (32)23. Each of the 32 attention heads processes a 128-dimensional slice of the full 4096-dimensional embedding, allowing the model to focus on different aspects of the input in parallel4
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values, history_token_ids

        evicted_tokens_index_start = self.start_size
        evicted_tokens_index_end = seq_len - self.recent_size + num_coming
        evicted_tokens_ids = history_token_ids[
            evicted_tokens_index_start:evicted_tokens_index_end
        ]
        evicted_tokens_text = " ".join(
            tokenizer.decode(
                evicted_tokens_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )
        history_token_ids = (
            history_token_ids[0:evicted_tokens_index_start]
            + history_token_ids[evicted_tokens_index_end:]
        )
        # print()
        # print()
        # print("-" * 200)
        # print(">>> Evicting the following text")
        # print()
        # print(evicted_tokens_text)
        # chunks = evicted_tokens_text.split(".")
        # print('evict:')
        # print(evicted_tokens_text)
        # chunks = re.findall(r'.*?[.!?\n]', evicted_tokens_text)
        # chunks = re.split(r'(?<=[.!?])|(?=(USER:|CONTEXT:|ASSISTANT:))', evicted_tokens_text)
        # chunks = re.split(r'(?<=[.|!|?|USER:|CONTEXT:|ASSISTANT:])', evicted_tokens_text)
        chunks = re.split(r'(?<=[.!?\n])', evicted_tokens_text)

        chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    
        embedded_chunks = torch.cat(
            [embed_text(chunk, model, tokenizer) for chunk in chunks], dim=0
        )
        # insert chunks in vector db
        # print()
        index.add(embedded_chunks)
        with open("data/embeddings.txt", "a") as file:
            file.writelines([chunk.strip("\n") + "\n" for chunk in chunks])
        # print(
        #     ">>> Inserted {} chunks in the db. Total entries: {}".format(
        #         len(embedded_chunks), index.ntotal
        #     )
        # )
        # print("-" * 200)
        # print()
        # print()

        new_past_key_values = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
        return new_past_key_values, history_token_ids

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
