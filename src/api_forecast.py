from fastapi import FastAPI
from pydantic import BaseModel
import torch
from chronos import BaseChronosPipeline

app = FastAPI(title="API Forecast de Demanda - Chronos-Bolt")
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)

class Solicitud(BaseModel):
    tienda: int
    historia: list[float]      # ventas semanales, de la más antigua a la más reciente
    horizonte: int = 12

@app.post("/forecast")
def forecast(req: Solicitud):
    q, _ = pipeline.predict_quantiles(inputs=torch.tensor(req.historia), prediction_length=req.horizonte,
                                      quantile_levels=[0.1, 0.5, 0.9])
    return {"tienda": req.tienda, "p10": q[0, :, 0].tolist(), "p50": q[0, :, 1].tolist(), "p90": q[0, :, 2].tolist()}