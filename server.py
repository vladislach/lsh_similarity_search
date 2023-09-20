from fastapi import FastAPI
from similarity_search import SimilaritySearch

app = FastAPI()

@app.get("/similarity_search/{smiles}")
def sim_search(smiles: str):
    searcher = SimilaritySearch()
    searcher.connect_and_load()
    return searcher.find_similar_molecules(smiles, threshold=0.3)