class ProteinTokenizer():
    ALL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTUVWXY"
    ADDITIONAL_TOKENS = ("&", "^", "$", "_")
    # & = OTHER, ^ = START, $ = END, _ = PAD
    ALL_TOKENS = tuple(ALL_AMINO_ACIDS) + ADDITIONAL_TOKENS
    ALL_TOKENS_SET = tuple(ALL_AMINO_ACIDS) + ADDITIONAL_TOKENS
    vocab_size = len(ALL_TOKENS)

    token_to_id = {token:i for i, token in enumerate(ALL_TOKENS)}
    id_to_token = {i:token for i, token in enumerate(ALL_TOKENS)}

    mask_token_id = ALL_TOKENS.index("X")
    pad_token_id = ALL_TOKENS.index("_")

    @classmethod
    def tokenize(cls, seq: str):
        # note: attention_mask will have 1 on END token
        # DiffusionBERT's attention_mask has 0 on END token
        return [cls.token_to_id[c] for c in f"^{seq}$"]

    @classmethod
    def untokenize(cls, seq: list):
        return "".join(map(lambda i: cls.id_to_token[i], seq))

    @classmethod
    def is_valid_seq(cls, seq):
        for token in seq:
            if not cls.is_valid_token(token):
                return False
        return True

    @classmethod
    def is_valid_token(cls, token):
        return token in cls.ALL_TOKENS_SET
