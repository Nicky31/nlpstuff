#!/bin/python3

import os
import sys
import argparse
import plac
import gensim
from dataclasses import replace
from paths import DATASETS, MODELS, Dataset
from training import train_model, load_model, SUPPORTED_MODDELS
from preprocessing import get_training_file

def training_stage(action, args):
    try:
        dataset = DATASETS[args.dataset]
    except KeyError:
        if not os.path.exists(args.dataset):
            raise Exception("Please specify a valid dataset id or path.")
        filename = os.path.split(args.dataset)[-1]
        dataset = Dataset(
            filename=filename,
            filepath=args.dataset,
            lang=args.language
        )
        
    
    if action == "preprocess":
        if args.out:
            dataset = dataset._replace(preprocessed_filepath=args.out)

        get_training_file(
            dataset=dataset,
            use_cache=False,
            lemmatize=False,
            remove_stopwords=True,
            progressbar=False if args.no_pb else True,
            tokenizer=args.tokenizer,
            workers=args.workers,
            chunk_size=args.chunk_size
        )
    elif action == "train":
        train_model(
            dataset=dataset,
            model_type=args.model,
            use_cache=True,
            dims=args.dims,
            window=args.window,
            workers=args.workers,
            model_filepath=args.out
        )
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    # Common opts
    parser.add_argument("--no-pb", help="Disable progress bar", action="store_true")
    parser.add_argument("--workers", help="Number of workers for preprocessing & training", default=4)

    existing_datasets = list(DATASETS.keys())

    # preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Training corpus preprocessing command')    
    preprocess_parser.add_argument("dataset", help="Dataset id/path to preprocess. \n Available datasets : {}".format(existing_datasets))
    preprocess_parser.add_argument("-l", "--language", help="fr, en, es, ...")
    preprocess_parser.add_argument("-o", "--out", help="Output preprocessed file path. Defaults to DATASET_DIR")
    preprocess_parser.add_argument("-t", "--tokenizer", choices=["spacy", "nltk"], help="Set a specific tokenizer", default="nltk")
    preprocess_parser.add_argument("--chunk-size", type=int, help="Size of dataset chunks distributed across the workers (bytes).", default=1024*256)

    # train command
    train_parser = subparsers.add_parser('train', help='Model training command')    
    train_parser.add_argument("dataset", help="Dataset id/path to train with. \n Available datasets : {}".format(existing_datasets))
    train_parser.add_argument("model", help="Algorithm to train with. \n Choose between {}".format(SUPPORTED_MODDELS))
    train_parser.add_argument("-o", "--out", help="Output trained model file path. Defaults to MODELS_DIR")
    train_parser.add_argument("-w", "--window", help="Window size", default=4)
    train_parser.add_argument("-d", "--dims", help="Vectors dimensions number", default=300)

    args = parser.parse_args()
    if args.action is None:
        print("Please specify a command between (preprocess, train).", file=sys.stderr)
        sys.exit(1)

    # try:
    if args.action in ("preprocess", "train"):
        training_stage(args.action, args)
    # except Exception as e:
        # print("An exception occured : {}".format(e), file=sys.stderr)

