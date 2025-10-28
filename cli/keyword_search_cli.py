#!/usr/bin/env python3

import argparse
import json
import os
import string
from nltk.stem import PorterStemmer
import InvertedIndex
import math


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    #step 1
    idf_parser = subparsers.add_parser("idf", help="Build inverted index")
    idf_parser.add_argument("term", type=str, help="IDF Operation")

    args = parser.parse_args()
    here = os.path.dirname(__file__)
    path_movies = os.path.join(here, "..", "data", "movies.json")
    path_stopwords = os.path.join(here,"..","data","stopwords.txt")
    punct_table = str.maketrans('', '', string.punctuation)
    stemmer = PorterStemmer()

    idx = InvertedIndex.InvertedIndex()

    with open(path_stopwords,"r") as f:
        stopwords = f.read()
    stopwords_set= set(stopwords.splitlines())
    
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

            val_count = idx.get_tf(args.doc_id,args.term,punct_table,stopwords_set,stemmer)

            print(f"doc ID:{args.doc_id}: {val_count}")

        case "build":
            
            movies = load_movies(path_movies)
            idx.build(movies["movies"],punct_table,stopwords_set,stemmer)
            idx.save()

            print(len(idx.docmap))

        case _:
            parser.print_help()

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