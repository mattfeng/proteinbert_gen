import torch
import pickle

def create_word_freq_tensor(word_freq_pkl_file: str, tokens: tuple):
    """
    Parameters:
        - word_freq_pkl_file:
            path to pickle file containing token->count mappings
        - tokens:
            tuple of all tokens
    """

    with open(word_freq_pkl_file, "rb") as f:
        word_freq_dict = pickle.load(f)

    wf_tensor = torch.zeros((len(tokens),), dtype=torch.int64)

    for i, tok in enumerate(tokens):
        wf_tensor[i] = word_freq_dict[tok]

    return wf_tensor
