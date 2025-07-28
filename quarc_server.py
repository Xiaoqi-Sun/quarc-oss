import argparse
import copy
import os
import sys
import datetime
import traceback
import uvicorn
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
from loguru import logger
from typing import Optional

from quarc_predictor import QuarcPredictor
import quarc_parser

app = FastAPI()
router = APIRouter()

base_response = {"status": "FAIL", "error": "", "results": []}


class RequestBody(BaseModel):
    smiles: list[str]
    top_k: Optional[int] = Field(10, ge=1, le=80, description="Number of top predictions")


@router.post("/condition_prediction")
def quarc_prediction_service(request_json: RequestBody):
    response = copy.deepcopy(base_response)

    try:
        results = predictor.predict_batch(request_json.smiles, top_k=request_json.top_k)

        response["results"] = results
        response["status"] = "SUCCESS"
        return response

    except Exception:
        response["error"] = (
            f"Error during QUARC prediction, traceback: " f"{traceback.format_exc()}"
        )
        traceback.print_exc()
        return response


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Condition recommender is running",
        "model_loaded": True if predictor is not None else False,
    }


app.include_router(router)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("quarc_server")
    quarc_parser.add_predict_opts(parser)
    quarc_parser.add_server_opts(parser)
    quarc_parser.add_data_opts(parser)

    args, unknown = parser.parse_known_args()

    # create logger
    os.makedirs("./logs/server", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    log_file = f"./logs/server/quarc_server.{dt}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    logger.add(log_file, level="INFO")

    # set up model
    predictor = QuarcPredictor(args)

    # start running
    uvicorn.run(app, host=args.server_ip, port=args.server_port)

    # python quarc_server.py --config-path=configs/gnn_pipeline.yaml
