from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from sentence_transformers import SentenceTransformer

# -------- Schemas --------
class EmbedRequest(BaseModel):
    instruction: str = Field(
        default="Represent the document for retrieval: ",
        description="INSTRUCTOR models expect an instruction paired with each text."
    )
    texts: List[str] = Field(..., description="Texts to embed.")
    normalize: bool = Field(default=True, description="L2-normalize embeddings.")
    batch_size: Optional[int] = Field(default=32, ge=1, description="Encode batch size.")

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

# -------- App --------
app = FastAPI(title="Instructor-XL Embedding API", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("/models/hkunlp__instructor-xl", device=device)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": device}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    pairs = [[req.instruction, t] for t in req.texts]
    embs = model.encode(
        pairs,
        convert_to_numpy=True,
        normalize_embeddings=req.normalize,
        batch_size=req.batch_size,
        show_progress_bar=False,
    ).tolist()
    return EmbedResponse(embeddings=embs)
