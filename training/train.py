import os
from preprocessing import get_training_file
from paths import Model
from gensim.models import Word2Vec, FastText

def train_model(dataset, model_type, use_cache=True, dims=300, window=4, min_count=4, workers=4, model_filepath=None):
    corpus_file = get_training_file(dataset, use_cache)
    if model_type == "word2vec":
        model = Word2Vec(
            corpus_file=corpus_file,
            size=dims,
            window=window,
            min_count=min_count,
            workers=workers
        )
        # Model params identification string, will be appended in output filename
        model_params = "cbow_w{}_dims{}".format(window, dims)
    elif model_type == "fastText":
        model = FastText(
            corpus_file=corpus_file,
            size=dims,
            window=window,
            min_count=min_count,
            workers=workers
        )
        # Model params identification string, will be appended in output filename
        model_params = "cbow_w{}_dims{}".format(window, dims)        
    else:
        raise Exception("Unknown model " + model)

    print("Model trained.")
    if model_filepath is None:
        model_filepath = Model(
            model_type=model_type,
            model_params=model_params,
            lang=dataset.lang,
            dataset=dataset.dataset
        ).filepath
    model.save(model_filepath)
    print("Model saved to '{}'".format(model_filepath))
    return model, model_filepath

def load_model(dataset):
    model = gensim.models.KeyedVectors.load_word2vec_format(name, binary=True)
    return model
