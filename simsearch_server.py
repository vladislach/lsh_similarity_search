import argparse
import copy
import logging
import os
import sys
import traceback
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from fastapi.routing import APIRouter
from pydantic import BaseModel
from rdkit import RDLogger
from typing import List

from similarity_search import SimilaritySearch

app = FastAPI()
router = APIRouter()

base_response = {
    "status": "FAIL",
    "error": "",
    "results": []
}

def parse_args():
    parser = argparse.ArgumentParser("simsearch_server")
    parser.add_argument("--server_ip", help="Server IP to use", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", help="Server port to use", type=int, default=9601)
    parser.add_argument("--log_file", help="Log file", type=str, default="simsearch_server")

    return parser.parse_args()


class RequestBody(BaseModel):
    smiles: List[str]


@app.get("/similarity_search/{smiles}")
def sim_search(smiles: str):
    return searcher.find_similar_molecules(smiles, threshold=0.3)

app.include_router(router)


if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs(f"./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}.log")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # set up model
    searcher = SimilaritySearch()
    searcher.connect_and_load()

    # start running
    uvicorn.run(app, host=args.server_ip, port=args.server_port)
