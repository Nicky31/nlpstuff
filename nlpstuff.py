#!/bin/python3

import os
import sys
import argparse
import plac
import gensim
from dataclasses import replace
from config import DATASETS, MODELS, Resource
from training import train_model, load_model
from preprocessing import get_training_file

def training_stage(action, args):
    try:
        dataset = DATASETS[args.resource]
    except KeyError:
        if not os.path.exists(args.resource):
            raise Exception("Please specify a valid dataset id or path.")
        filename = os.path.split(args.resource)[-1]
        dataset = Resource(
            filename=filename,
            filepath=args.resource,
            dataset=filename,
            lang=args.language
        )
        
    if args.training_out:
        dataset = dataset._replace(cached_training_filepath=args.training_out)
    
    if action == "preprocess":
        training_file = get_training_file(
            dataset,
            use_cache=False,
            lemmatize=False,
            remove_stopwords=True,
            progressbar=False if args.no_pb else True,
            tokenizer=args.tokenizer
        )
        print("Training file generated in '{}'".format(training_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preprocess", "train"])
    existing_datasets = list(DATASETS.keys())
    parser.add_argument("resource", help="Dataset id/path if preprocessing / training, otherwise model id / path. \n Available datasets : {}".format(existing_datasets))
    parser.add_argument("-l", "--language", help="fr, en, es, ...")
    parser.add_argument("--training-out", help="Filepath of generated training file. Defaults to DATASET_DIR config")
    parser.add_argument("-t", "--tokenizer", choices=["spacy", "nltk"], help="Set a specific tokenizer", default="nltk")

    parser.add_argument("-m", "--model", help="Model ID")
    parser.add_argument("-w", "--window", help="Window size")
    parser.add_argument("--no-pb", help="Disable progress bar", action="store_true")
    parser.add_argument("--workers", help="Number of workers for preprocessing & training")
    args = parser.parse_args()

    try:
        if args.action in ("preprocess", "train"):
            training_stage(args.action, args)
    except Exception as e:
        print("An exception occured : {}".format(e), file=sys.stderr)

