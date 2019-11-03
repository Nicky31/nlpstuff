from .train import train_model, load_model

SUPPORTED_MODDELS = [
    "word2vec",
    "fastText"
]

__all__ = ["train_model", "load_model", "SUPPORTED_MODDELS"]