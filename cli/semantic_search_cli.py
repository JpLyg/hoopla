import argparse
from lib import semantic_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Get BM25 IDF score for a given term")
    
    args = parser.parse_args()

    

    match args.command:
        case "help":
            parser.print_help()
        case "verify":
                semantic_search.verify_model()

if __name__ == "__main__":
    main()