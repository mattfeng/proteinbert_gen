# data constants
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ALL_AAS_SET = set(ALL_AAS)
ADDITIONAL_TOKENS = ('<OTHER>', '<START>', '<END>', '<PAD>')
ALL_TOKENS = tuple(ALL_AAS) + ADDITIONAL_TOKENS
VOCAB_SIZE = len(ALL_TOKENS)

assert VOCAB_SIZE == 26, "vocab size should be 26"

# model constants
GO_ANN_SIZE = 8943

# training constants
BATCH_SIZE = 32
