# nlpstuff

As its name suggests, nlpstuff gathers common NLP tasks in a single tool while providing differents implementations behind a standardized CLI interface.  

## Features

* parallelized **corpus preprocessing**
    * Tokenization (nltk, spacy)
    * stopwords removal (nltk, spacy)
    * lemmatization (spacy)
* **model training**
    * word2vec (cbow) 
    * fastText (cbow)

### Preprocessing

Given a fr_imdb dataset registered in paths.py :
```
./nlpstuff.py preprocess fr_imdb 
```
By default, only tokenization and stopwords removal are applied.  
Preprocessed file will be stored in a *preprocessed* subfolder of *DATASETS_DIR* env variable.  
This can be changed using **--output** :
```
./nlpstuff.py preprocess fr_imdb -o /tmp/fr_imdb.preprocessed
```

If input dataset has not been registered in your paths.py, you can indicate a filepath as well :
```
./nlpstuff preprocess /tmp/datasets/fr_imdb.txt -o /tmp/datasets/fr_imdb.preprocessed
```

**Other parameters :**
* *--tokenizer* / *-t* {tokenizer} where tokenizer is one of {nltk, spacy}
    * Defaults : **nltk**, since it's the fastest one
* *--lang* / *-l* {lang} where lang is the dataset's 2-letters language {fr, en, ...}
    * Defaults : extracted from dataset filename first 2 letters
* *--no-pb* disable tqdm progressbar
* *--chunk-size* modify preprocessed chunks size
    * Defaults: **256 KB**
* *--workers* change number of used processes 
    * Defaults : **4**
    * Each worker opens input file and read, preprocess and write at least *--chunk-size* per iteration


### Training

Say we want to train *fr_smallimdb* dataset with *word2vec* :
```
./nlpstuff.py train fr_smallimdb word2vec
```

Any dataset's preprocessed version found in the regular *preprocessed* subfolder of *DATASETS_DIR* will be used. As a result, outdated preprocessed files should be manually removed.  
  
Like for preprocessed files, output model location defaults to *MODELS_DIR* env variable but can be changed with *-o*.  
Default filename is built using main settings from your training. As an example, last command would generate a model *fr_smallimdb_word2vec_cbow_w4_dims300.bin*.  
This default formatting helps to keep track of differences between available models.  

**Other parameters :**
* *--window* / *-w* window size
    * Defaults **4**
* *--dimensions* / *-d* vectors dimension
    * Defaults **300**
