import os, pickle


class InvertedIndex:
    
    #step 1 and 2
    def __init__(self):
        self.index = {}
        self.docmap = {}
    #step 3
    def __tokenize(self, content: str,punct_table,stopwords_set,stemmer):
        content = content.lower()
        content = content.translate(punct_table)
        content = content.strip()

        content = content.split()
        content = list(filter(lambda x: x.strip(), content))
        content = list(filter(lambda x: x not in stopwords_set, content))
        content = [stemmer.stem(t) for t in content]

        return content

        #return text.lower().split()

    def __add_document(self, doc_id,text,punct_table,stopwords_set,stemmer):
        for tok in self.__tokenize(text,punct_table,stopwords_set,stemmer):
            if tok not in self.index:
                self.index[tok] = set()
            self.index[tok].add(doc_id)
    #step 4
    def get_document(self,term):
        tok = term.lower()
        ids = self.index.get(tok,set())
        return sorted(ids)
    #step 5
    def build(self, movies,punct_table,stopwords_set,stemmer):
        for m in movies:
            
            doc_id = m["id"]
            
            self.docmap[doc_id] = m["title"]
            text = f"{m['title']} {m['description']}"
            
            self.__add_document(doc_id, text,punct_table,stopwords_set,stemmer)
    
    #step 6
    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)