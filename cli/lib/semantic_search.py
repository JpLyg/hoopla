from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        

def verify_model():
    instance = SemanticSearch() 
    print(f"Model loaded: {instance.model}")
    print(f"Max sequence length: {instance.model.max_seq_length}")