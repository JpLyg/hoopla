import os, pickle
from collections import Counter
from nltk.stem import PorterStemmer
import math


class InvertedIndex:
    
    
    def __init__(self,punc_tbl,stopwords,stems):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}



        self.stopwords = stopwords
        self.punc_tbl = punc_tbl
        self.stems = stems
    
    def __tokenize(self, content: str):
        content = content.lower()
        content = content.translate(self.punc_tbl)
        content = content.strip()

        content = content.split()
        content = list(filter(lambda x: x.strip(), content))
        content = list(filter(lambda x: x not in self.stopwords, content))
        content = [self.stems.stem(t) for t in content]

        return content

    
    def get_tf(self, doc_id, term) -> int:

        term = self.__tokenize(term)
        if len(term) > 1:
            raise ValueError
        
        if doc_id not in self.term_frequencies:
            return 0

        if term[0] not in self.term_frequencies[doc_id]:
            return 0
        
        return self.term_frequencies[doc_id][term[0]]
        

    def __add_document(self, doc_id,text):

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        tokens = self.__tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for tok in tokens:
            if tok not in self.index:
                self.index[tok] = set()
            self.index[tok].add(doc_id)
            self.term_frequencies[doc_id][tok] += 1


    def get_document(self,term):
        tok = term.lower()
        ids = self.index.get(tok,set())
        return sorted(ids)

    def build(self, movies):
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m["title"]
            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
            try:
                 with open("cache/doc_lengths.pkl", "rb") as f:
                    self.doc_lengths = pickle.load(f)
            except FileNotFoundError:
                self.doc_lengths = {}

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)

        term_list = self.__tokenize(term)
        if len(term_list) != 1:
            raise ValueError ("expected a single token")
        
        posting = self.index.get(term_list[0], set())
        df = len(posting)

        bm25 = math.log((N - df + 0.5)/(df+0.5)+1)

        return bm25

    def get_bm25_tf(self, doc_id: int, term: str, k1: float,b:float) -> float:
        tf = self.get_tf(doc_id,term)
        if tf == 0:
            return 0.0
        
        doc_len = self.doc_lengths.get(doc_id, 0)
        avg_len = self.__get_avg_doc_length()
        length_norm = 1.0
        if avg_len > 0:
            length_norm = 1 - b + b * (doc_len / avg_len)   

        bm25_saturation = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_saturation
    
    # python
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total = sum(self.doc_lengths.values())
        return total / len(self.doc_lengths)

    