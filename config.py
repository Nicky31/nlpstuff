import os
from dataclasses import dataclass
from collections import namedtuple

MODELS_DIR = os.getenv("MODELS_DIR", "/home/martin/Medias/ai/models/")
DATASETS_DIR = os.getenv("DATASETS_DIR", "/home/martin/Medias/ai/datasets/")

resource_fields = {
    "filename": "",
    "model": "",
    "filepath": "",
    "dataset": "",
    "lang": "",
    "cached_training_filepath": ""
}
Res = namedtuple("Resource", list(resource_fields.keys()))
def Resource(**kwargs):
    if "filepath" in kwargs:
        print(kwargs["filename"])
    data = {
        **resource_fields,
        **kwargs
    }
    if not data["lang"]:
        data["lang"] = data["filename"][0:2]

    if not data["filepath"]:
        if not data["model"]:
            base_dir = DATASETS_DIR
        else:
            base_dir = os.path.join(MODELS_DIR, data["model"])

        data["filepath"] = os.path.join(base_dir, data["lang"], data["filename"])

    if not data["cached_training_filepath"]:
        split = os.path.split(data["filepath"])
        data["cached_training_filepath"] = os.path.join(split[0], "__cache__",  data["filename"])  

    return Res(**data)

def get_resources(*resources):
    return {
        os.path.splitext(res.filename)[0]: res for res in resources
    }



MODELS = {
    "word2vec": get_resources(
        Resource(filename="en_negative_googlenews_300.bin", model="word2vec"),
        Resource(filename="fr_negative_googlenews_300.bin", model="word2vec"),
        Resource(filename="fr_cbow_imdb_300.bin", model="word2vec")
    ),

    "fastText": get_resources(
        Resource(filename="fr_cbow_cc_wiki_300.bin", model="fastText"),
        Resource(filename="en_cbow_cc_wiki_300.bin", model="fastText")
    ),
}

DATASETS = get_resources(
    Resource(filename="fr_imdb.txt", dataset="imdb"),
    Resource(filename="en_imdb.txt", dataset="imdb"),
)