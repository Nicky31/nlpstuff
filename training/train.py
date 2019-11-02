import os
from preprocessing import get_training_file
from config import DATASETS, MODELS, Resource
from gensim.models import Word2Vec

def train_model(dataset, model_type, use_cache=True, size=300, window=4, min_count=4, workers=4):
    dataset = DATASETS[dataset]

    corpus_file = get_training_file(dataset, use_cache)
    if model_type == "word2vec":
        model = Word2Vec(
            corpus_file=corpus_file,
            size=size,
            window=window,
            min_count=min_count,
            workers=workers         
        )
        model_filename = "{}_cbow_{}_{}.bin".format(dataset.lang, dataset.dataset, window)
    else:
        raise Exception("Unknown model " + model)

    print("Model trained.")
    model_filepath = Resource(filename=model_filename, model=model_type).filepath
    model.save(model_filepath)
    print("Model saved to '{}'".format(model_filepath))
    return model

def load_model(dataset):
    model = gensim.models.KeyedVectors.load_word2vec_format(name, binary=True)
    return model
