import os, pickle
from collections import Counter


class InvertedIndex:
    
    
    def __init__(self):
        self.index = {}
        self.docmap = {}
        #step 1
        self.term_frequencies = {}
    
    def __tokenize(self, content: str,punct_table,stopwords_set,stemmer):
        content = content.lower()
        content = content.translate(punct_table)
        content = content.strip()

        content = content.split()
        content = list(filter(lambda x: x.strip(), content))
        content = list(filter(lambda x: x not in stopwords_set, content))
        content = [stemmer.stem(t) for t in content]

        return content

    #step 4
    def get_tf(self, doc_id, term,punct_table,stopwords_set,stemmer ) -> int:

        term = self.__tokenize(term,punct_table,stopwords_set,stemmer)

        if len(term) > 1:
            raise ValueError
        
        if doc_id not in self.term_frequencies:
            return 0

        if term[0] not in self.term_frequencies[doc_id]:
            return 0
        
        return self.term_frequencies[doc_id][term[0]]
        




    def __add_document(self, doc_id,text,punct_table,stopwords_set,stemmer):

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for tok in self.__tokenize(text,punct_table,stopwords_set,stemmer):
            if tok not in self.index:
                self.index[tok] = set()
            self.index[tok].add(doc_id)

            self.term_frequencies[doc_id][tok] += 1

    def get_document(self,term):
        tok = term.lower()
        ids = self.index.get(tok,set())
        return sorted(ids)

    def build(self, movies,punct_table,stopwords_set,stemmer):
        for m in movies:
            
            doc_id = m["id"]
            
            self.docmap[doc_id] = m["title"]
            text = f"{m['title']} {m['description']}"
            
            self.__add_document(doc_id, text,punct_table,stopwords_set,stemmer)

    

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)