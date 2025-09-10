from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from sentence_transformers import CrossEncoder

# -------- Schemas --------
class RerankRequest(BaseModel):
    query: str = Field(..., description="User query.")
    candidates: List[str] = Field(..., description="Candidate passages to score.")
    batch_size: Optional[int] = Field(default=32, ge=1, description="Predict batch size.")

class RerankResponse(BaseModel):
    scores: List[float]  # parallel to candidates order

# -------- App --------
app = FastAPI(title="BGE Reranker Large API", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder("/models/BAAI__bge-reranker-large", device=device)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": device}

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    pairs = [(req.query, c) for c in req.candidates]
    scores = model.predict(pairs, batch_size=req.batch_size).tolist()
    return RerankResponse(scores=scores)
