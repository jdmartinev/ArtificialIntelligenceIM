import os

def get_corpus(path: str = "corpus.txt") -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    # fallback to built-in corpus if no file found
    print(f"[data] '{path}' not found, using built-in corpus.")
    return CORPUS.strip()
 
 
if __name__ == "__main__":
    corpus = get_corpus()
    print(f"Corpus length: {len(corpus)} characters")
    print(f"First 200 chars:\n{corpus[:200]}")