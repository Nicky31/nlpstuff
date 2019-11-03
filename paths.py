import os
from dataclasses import dataclass
from collections import namedtuple

MODELS_DIR = os.getenv("MODELS_DIR", "/home/martin/Medias/ai/models/")
DATASETS_DIR = os.getenv("DATASETS_DIR", "/home/martin/Medias/ai/datasets/")

PREPROCESSED_DIRNAME = "__preprocessed__"
MODEL_FILENAME_FORMAT = "{lang}_{dataset}_{model_type}_{model_params}.bin"
# for lang & dataset extraction, raw dataset filenames must be the same as preprocessed ones :
PREPROCESSED_FILENAME_FORMAT =  "{lang}_{dataset}.txt"

"""
    Model & Dataset object fields
    Both object filenames begins with lang & source dataset identifers, allowing us to easily retrieve informations
    As a result, in case of user-defined filenames, certain functions may need explicit settings definition

    Dataset filename default formatting : 
        "fr_imdb.txt" -> {lang}_{dataset}.txt

    Model filename default formatting : 
        "fr_imdb_word2vec_w4_300_cbow.bin" ->  {lang}_{dataset}_{model_type}_{model_params}.bin
"""
resource_fields = {
    # Common fields
    "filename": "", # Dataset or model filename, with extension
    "filepath": "", # full filepath
    "dataset": "", # dataset name, prefixed with lang
    "lang": "", # source dataset language (extracted from filename if not specified)
    "preprocessed_filepath": "", # preprocessed training file path

    # Model-only fields
    "model_type": "", # model type: word2vec, fastText, ...
    "model_params": "", # custom model settings
}
Res = namedtuple("Resource", list(resource_fields.keys()))

# Build default common fields from filename
# If not provided, raises an exception
def get_res_paths(base_dir, **kwargs):
    data = {
        **resource_fields,
        **kwargs
    }

    split = os.path.splitext(data["filename"])[0].split("_")

    if not data["lang"]:
        data["lang"] = split[0]
    if not data["filepath"]:
        data["filepath"] = os.path.join(base_dir, data["lang"], data["filename"])
    if not data["dataset"]:
        data["dataset"] = split[1]
    if not data["preprocessed_filepath"]:
        data["preprocessed_filepath"] = os.path.join(
            DATASETS_DIR,
            data["lang"],
            PREPROCESSED_DIRNAME, 
            PREPROCESSED_FILENAME_FORMAT.format(**data)
        )

    return data

"""
    Dataset paths construction

    Parameters :
        required :filename -> filename with extension and prefixed with lang (i.e fr_imdb.txt)

        optional :lang -> required if not prefixed in filename (i.e fr_imdb.txt)
        optional :filepath -> overrides filepath automatically constructed from DATASETS_DIR
        optional :preprocessed_filepath -> overrides preprocessed filepath automatically constructed from DATASETS_DIR
"""
def Dataset(**kwargs):
    fields = get_res_paths(base_dir=DATASETS_DIR, **kwargs)
    return Res(**fields)


"""
    Model paths construction

    Read-mode parameters : extract metadata from filename formatting
        required :filename -> filename with extension

    Write-mode parameters : build filename from metadata
        required :model_type -> word2vec, fastText, ...
        required :lang -> fr, en, ...
        required :dataset -> imdb, wikipedia, ...
        required :model_params -> model-specific settings ; cbow_window4, ngram_dims300, ...

"""
def Model(**kwargs):
    try:
        fields = get_res_paths(base_dir=MODELS_DIR, **kwargs)
    except: # Allow model construction without filename/filepath for output path construction
        fields = resource_fields

    # Read mode : extract model informations from given filename
    if fields["filename"]:
        try:
            # remove extension .bin before splitting
            split = os.path.splitext(fields["filename"])[0].split("_")
            fields["model_type"] = split[2]
            fields["model_params"] = "_".join(split[3:]) if len(split) > 3 else ""
        except:
            print("Warning : unable to extract information from model filename '{}'".format(fields["filename"]))

    
    # write mode : build filename/path from other params
    if not fields["filename"] or not fields["filepath"]:
        fields["filename"] = MODEL_FILENAME_FORMAT.format(**kwargs)
        fields["filepath"] = os.path.join(MODELS_DIR, kwargs["lang"], fields["filename"])

    return Res(**fields)

def get_resources(*resources):
    return {
        os.path.splitext(res.filename)[0]: res for res in resources
    }


MODELS = {
    "word2vec": get_resources(
        Model(filename="en_googlenews_word2vec_dims300.bin"),
    ),

    "fastText": get_resources(
        Model(filename="fr_ccwiki_fastText_cbow_w300.bin"),
        Model(filename="en_ccwiki_fastText_dims300.bin")
    ),
}

DATASETS = get_resources(
    Dataset(filename="fr_imdb.txt"),
    Dataset(filename="fr_smallimdb.txt"),
    Dataset(filename="en_imdb.txt"),
)