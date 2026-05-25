"""FastAPI service for motor insurance pure premium prediction.

App flow:
1) Load CatBoost models at startup.
2) Validate input payload through Pydantic schema.
3) Build a one-row dataframe for model scoring.
4) Predict frequency and severity, then combine into pure premium.
"""

#%% required libraries
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4
import logging

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from catboost import CatBoostRegressor
import pandas as pd
from schemas import Insured, PredictionResponse
from steps.predict import Predictor
import uvicorn

logger = logging.getLogger("pricing_api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources and keep app alive even if model loading fails."""
    app.state.model_freq = None
    app.state.model_sev = None
    app.state.models_ready = False
    app.state.model_load_error = None

    try:
        model_freq = CatBoostRegressor()
        model_freq.load_model('models/frequency_model.cbm')
        model_sev = CatBoostRegressor()
        model_sev.load_model('models/severity_model.cbm')

        app.state.model_freq = model_freq
        app.state.model_sev = model_sev
        app.state.models_ready = True
        logger.info("event=models_loaded status=ok")
    except Exception as exc:
        app.state.model_load_error = str(exc)
        logger.exception("event=models_loaded status=failed")

    yield


# Single FastAPI application instance exposed by uvicorn (`app:app`).
app = FastAPI(lifespan=lifespan)

# Factory helper used by FastAPI dependency injection.
# It wraps a model with the right prediction mode.
def get_frequency_predictor(request: Request) -> Predictor:
    if not request.app.state.models_ready or request.app.state.model_freq is None:
        raise HTTPException(status_code=503, detail="Frequency model not available")
    return Predictor(request.app.state.model_freq, 'frequency')  # type: ignore[arg-type]


def get_severity_predictor(request: Request) -> Predictor:
    if not request.app.state.models_ready or request.app.state.model_sev is None:
        raise HTTPException(status_code=503, detail="Severity model not available")
    return Predictor(request.app.state.model_sev, 'severity')  # type: ignore[arg-type]


def model_to_dict(model: BaseModel) -> dict:
    """Support both Pydantic v2 (`model_dump`) and legacy v1 (`dict`)."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.get("/", response_model=dict)
async def read_root():
    """Basic API info endpoint."""
    return {
        "service": "pricing-api",
        "docs": "/docs",
        "health": "/health",
        "readiness": "/ready",
    }


@app.get("/health", response_model=dict)
async def health():
    """Liveness endpoint: process is running."""
    return {"status": "ok"}


@app.get("/ready", response_model=dict)
async def ready(request: Request):
    """Readiness endpoint: models are loaded and API can serve predictions."""
    if request.app.state.models_ready:
        return {"status": "ready"}
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "reason": request.app.state.model_load_error or "models_not_loaded",
        },
    )

# Main business endpoint: validate input, score both models, compose output.
@app.post("/predict/", response_model=PredictionResponse)
async def predict(
    request: Request,
    insured: Insured,
    predictor_freq: Predictor = Depends(get_frequency_predictor),
    predictor_sev: Predictor = Depends(get_severity_predictor)
):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    start = perf_counter()

    logger.info("event=predict_started request_id=%s", request_id)

    # Convert validated payload into a single-row dataframe.
    # CatBoost pipeline in `Predictor` expects tabular input.
    df_insured = pd.DataFrame([model_to_dict(insured)])

    # Required helper columns expected by training/prediction pipeline:
    # - `log_exposure` for frequency model baseline
    # - `ClaimNb` for severity model weighting path
    df_insured['log_exposure'] = 0.0
    df_insured['ClaimNb'] = 1

    try:
        # Score frequency and severity independently.
        pred_freq = predictor_freq.predict(df_insured)[0]
        pred_sev = predictor_sev.predict(df_insured)[0]

        # Actuarial composition rule: pure premium = expected frequency * expected severity.
        pure_premium = pred_freq * pred_sev

        latency_ms = (perf_counter() - start) * 1000
        logger.info(
            "event=predict_completed request_id=%s latency_ms=%.2f outcome=success",
            request_id,
            latency_ms,
        )

        # Return structured response documented by `PredictionResponse`.
        return PredictionResponse(Frequency=pred_freq, Severity=pred_sev, Pure_Premium=pure_premium)
    except Exception:
        latency_ms = (perf_counter() - start) * 1000
        logger.exception(
            "event=predict_completed request_id=%s latency_ms=%.2f outcome=error",
            request_id,
            latency_ms,
        )
        raise

#%% local entrypoint for direct execution (python app.py)
if __name__ == "__main__":
    uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )