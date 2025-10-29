#!/usr/bin/env python3

import argparse
import json
import os
import string
from nltk.stem import PorterStemmer
import InvertedIndex
import math


def main() -> None:
    BM25_K1 = 1.5
    BM25_B = 0.75
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser("idf", help="Build inverted index")
    idf_parser.add_argument("term", type=str, help="IDF Operation")

    tfidf_parser = subparsers.add_parser("tfidf", help="TFIDF Operation")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search Keyword")

    args = parser.parse_args()
    here = os.path.dirname(__file__)
    path_movies = os.path.join(here, "..", "data", "movies.json")
    path_stopwords = os.path.join(here,"..","data","stopwords.txt")

    punct_table = str.maketrans('', '', string.punctuation)
    stemmer = PorterStemmer()

    with open(path_stopwords,"r") as f:
        stopwords = f.read()
    stopwords_set= set(stopwords.splitlines())

    idx = InvertedIndex.InvertedIndex(punct_table,stopwords_set,stemmer)
    

    match args.command:
        case "search":
            try:
                idx.load()
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return     
            print("Searching for: " +args.query)

            results = set()
            
            for i in split_vals(translates(args.query,punct_table),stopwords_set,stemmer):
                results.update(idx.index.get(i, set()))
                
            results = sorted(results)[:5]
            print(results)

            for i in results:
                print (f"title: {idx.docmap[i]} id: {i}")

        case "bm25tf":
            try:
                idx.load()
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            bm25tf = bm25_tf_command(args.doc_id,args.term,BM25_K1,BM25_B,idx)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25idf":
            try:
                idx.load()
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            
            bm25idf= bm25_idf_command(idx,args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        
        case "tfidf":
            try:
                idx.load()
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            
            tf = idx.get_tf(args.doc_id,args.term)
            
            idf = idf_func(idx,args.term,stopwords,stemmer)
            

            tf_idf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "idf":
            try:
                idx.load()
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            
            docmap_count = len(idx.docmap)
            proper_term = split_vals(args.term,stopwords_set,stemmer)
            postings = idx.index.get(proper_term[0], set())
            term_doc_count = len(postings)
            idf_val = math.log((docmap_count+1)/(term_doc_count+1))

            
            print(f"Inverse document frequency of '{args.term}': {idf_val:.2f} | doc count: {docmap_count} | term doc count: {term_doc_count}")

        case "tf":
            try:
                idx.load()
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            val_count = idx.get_tf(args.doc_id,args.term)

            print(f"doc ID:{args.doc_id}: {val_count}")

        case "build":
            
            movies = load_movies(path_movies)
            idx.build(movies["movies"])
            idx.save()

            print(len(idx.docmap))

        case _:
            parser.print_help()

def bm25_tf_command(doc_id,term,k1,b,idx):
    bm25_saturation = idx.get_bm25_tf(doc_id,term,k1,b)
    return bm25_saturation

def bm25_idf_command(idx,term):
    bm25idf = idx.get_bm25_idf(term)
    return bm25idf

def idf_func(idx,term,stopwords,stemmer):
    docmap_count = len(idx.docmap)
    proper_term = split_vals(term,stopwords,stemmer)

    postings = idx.index.get(proper_term[0], set())
    term_doc_count = len(postings)

    return math.log((docmap_count+1)/(term_doc_count+1))

def translates(content, punc_vals):
    content = content.lower()
    content = content.translate(punc_vals)
    content = content.strip()
    return content

def split_vals(query,stopwords_set,stemmer):
    content_list = query.split()
    content_list = list(filter(lambda x: x.strip(), content_list))
    content_list = list(filter(lambda x: x not in stopwords_set, content_list))
    content_list = [stemmer.stem(t) for t in content_list]
    return content_list

def load_movies(path_movies):
    with open(path_movies, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    main()