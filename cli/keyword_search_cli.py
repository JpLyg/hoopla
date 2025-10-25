#!/usr/bin/env python3

import argparse
import json
import os
import string
from nltk.stem import PorterStemmer
import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

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
                print(idx.index.get("brave", set()))
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            
            print("Searching for: " +args.query)

            #data = load_movies(path_movies)
            results = set()
            

            for i in split_vals(translates(args.query,punct_table),stopwords_set,stemmer):
                results.update(idx.index.get(i, set()))
                #if len(results)>= 5:break
                
            results = sorted(results)[:5]
            print(results)

            for i in results:
                print (f"title: {idx.docmap[i]} id: {i}")
                
            '''for r in data["movies"]:
                for i in split_vals(translates(args.query,punct_table),stopwords_set,stemmer):
                    if i in split_vals(translates(r["title"],punct_table),stopwords_set,stemmer):
                        results.append(r)
                        break

            results.sort(key=lambda m: m["id"])'''
            
            
            

            

        case "build":
            
            movies = load_movies(path_movies)
            idx.build(movies["movies"],punct_table,stopwords_set,stemmer)
            idx.save()
            
            #print(docs)
            #print(f"First document for token 'merida' = {docs[0]}")
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