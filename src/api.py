import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/phishing_model.joblib"

# Carrega o modelo UMA vez, no startup (não a cada requisição!)
bundle = joblib.load(MODEL_PATH)
vectorizer = bundle["vectorizer"]
model = bundle["model"]

app = FastAPI(title="Phishing ML Comparison API", description="Compara 4 abordagens de detecção de phishing em URLs")

class URLRequest(BaseModel):
    url: str

@app.get("/")
def health():
    return {"status": "ok", "service": "phishing-ml-comparison"}


@app.post("/predict")
def predict(request: URLRequest):
    X = vectorizer.transform([request.url])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()
    return {
        "url": request.url,
        "label": label,
        "confidence": round(float(proba), 4),
    }