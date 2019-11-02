import unicodedata
import string
from importlib import import_module
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

punctuation_translation = str.maketrans(string.punctuation, " " * len(string.punctuation))
digit_translation = str.maketrans('', '', string.digits)

spacy_nlp = {}
def get_spacy_nlp(lang, disable=['parser', 'ner', 'tagger', 'textcat']):
     if lang in spacy_nlp:
          return spacy_nlp[lang]
     spacy_nlp[lang] = import_module(lang+"_core_news_sm").load(disable=disable)
     return spacy_nlp[lang]

class Preprocessing:
     """
          Corpus is a list of strings
     """
     def __init__(self, lang, remove_stopwords=True, lemmatize=True, tokenizer='nltk'):
          self.lang = lang
          if tokenizer == 'spacy':
               self.spacy = get_spacy_nlp(lang)
          elif tokenizer == 'nltk':
               self.stopwords = set(stopwords.words('french'))

          self.lemmatize = lemmatize
          self.remove_stopwords = remove_stopwords

          try:
               self.tokenize = {
                    "spacy": self.spacy_tokenize,
                    "nltk": self.nltk_tokenize
               }[tokenizer]
          except:
               raise Exception("Unknown tokenizer '{}'".format(tokenizer))

     """
          Lowercase, remove punctuation, digits and trailing withespaces from s
     """
     def clean(self, s):
          return s.lower()\
               .translate(punctuation_translation)\
               .translate(digit_translation)\
               .strip()

     """
          Tokenize, remove stop than lemmatize words
     """
     def spacy_tokenize(self, s, remove_stopwords=True, lemmatize=True):
          doc = self.spacy(s)
          if not remove_stopwords:
               return [(token.text if not lemmatize else token.lemma_) for token in doc]
          return [(token.text if not lemmatize else token.lemma_) for token in doc if not token.is_stop]               

     def nltk_tokenize(self, s, remove_stopwords=True, lemmatize=True):
          if not remove_stopwords:
               return word_tokenize(s)
          return [w for w in word_tokenize(s) if not w in self.stopwords]

     def __call__(self, sentences):
          return [
               "{}\n".format(" ".join(self.tokenize(
                    s=self.clean(s),
                    remove_stopwords=self.remove_stopwords,
                    lemmatize=self.lemmatize
               ))) for s in sentences
          ]
