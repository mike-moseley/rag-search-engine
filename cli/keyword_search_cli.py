import string
import json
import argparse
import inverted_index
from nltk.stem import PorterStemmer

stop_words_f = open("data/stopwords.txt")
stop_words = stop_words_f.read().splitlines()
punc_table = str.maketrans("", "", string.punctuation)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index")

    args = parser.parse_args()
    stemmer = PorterStemmer()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # print the search query here
            moviesFile = open("data/movies.json")
            movies = json.load(moviesFile)["movies"]
            stripped_query = args.query.lower().translate(punc_table)
            tokenized_query = tokenize(stripped_query)
            tokenized_query = [q for q in tokenized_query if q not in stop_words]
            stemmed_query = [stemmer.stem(q) for q in tokenized_query ]

            count = 0
            for movie in movies:
                stripped_movie_title = movie["title"].lower().translate(punc_table)
                tokenized_movie_title = tokenize(stripped_movie_title)
                tokenized_movie_title = [m for m in tokenized_movie_title if m not in stop_words]
                stemmed_movie_title = [stemmer.stem(m) for m in tokenized_movie_title]
                if count == 5:
                    return
                for query in stemmed_query:
                    if any(query in token for token in stemmed_movie_title):
                        count = count + 1
                        print(f'{count}. {movie["title"]}')
                        break
            pass
        case "build":
            ii = inverted_index.InvertedIndex()
            ii.build()
            ii.save()
            docs = ii.get_documents('merida')
            print(f"First document token 'merida' = {docs[0]}")
        case _:
            parser.print_help()

def tokenize(str):
    stemmer = PorterStemmer()
    stripped = str.lower().translate(punc_table)
    tokens = [s for s in stripped.split(" ") if s != ""]
    
    tokens = [q for q in tokens if q not in stop_words]
    return [stemmer.stem(q) for q in tokens ]

def load_movies():
    with open("data/movies.json") as f:
        return json.load(f)["movies"]

if __name__ == "__main__":
    main()
