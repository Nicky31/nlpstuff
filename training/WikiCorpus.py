from gensim.corpora.wikicorpus import WikiCorpus

class WikiCorpus:
    def __init__(self, wiki_dump_path, lang):
        logging.info('Parsing wiki corpus')
        self.wiki = WikiCorpus(wiki_dump_path)
        self.lang = lang

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            yield list(sentence)