"""FastAPI inference service. Start with: uvicorn src.service:app --host 0.0.0.0 --port 8000"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.modeling import load_model
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_time_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

app = FastAPI(title="Traffic Volume Predictor", version="1.0")

_model = None

_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Traffic Volume Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 540px; margin: 60px auto; background: #f4f6f8; }
    h1 { color: #2c3e50; font-size: 1.5rem; margin-bottom: 4px; }
    p.sub { color: #666; margin-top: 0; margin-bottom: 24px; font-size: 0.9rem; }
    .card { background: #fff; border-radius: 10px; padding: 28px 32px; box-shadow: 0 2px 8px rgba(0,0,0,.1); }
    label { display: block; margin-top: 14px; font-size: 0.85rem; color: #444; font-weight: bold; }
    input, select { width: 100%; padding: 8px 10px; margin-top: 4px; border: 1px solid #ccc;
                    border-radius: 6px; font-size: 0.95rem; box-sizing: border-box; }
    button { margin-top: 22px; width: 100%; padding: 11px; background: #2c7be5;
             color: #fff; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }
    button:hover { background: #1a5fbf; }
    #result { margin-top: 20px; padding: 16px; border-radius: 8px; text-align: center;
              font-size: 1.1rem; display: none; }
    .ok  { background: #e6f4ea; color: #2d6a4f; border: 1px solid #b7dfca; }
    .err { background: #fdecea; color: #a93226; border: 1px solid #f5b7b1; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Traffic Volume Predictor</h1>
    <p class="sub">Прогноз почасового трафика на I-94 (Minneapolis/St Paul)</p>

    <label>Дата и время
      <input type="datetime-local" id="date_time" value="2018-06-15T08:00">
    </label>
    <label>Температура (K)
      <input type="number" id="temp" value="295" step="0.1">
    </label>
    <label>Дождь за час (мм)
      <input type="number" id="rain_1h" value="0" step="0.1" min="0">
    </label>
    <label>Снег за час (мм)
      <input type="number" id="snow_1h" value="0" step="0.1" min="0">
    </label>
    <label>Облачность (%)
      <input type="number" id="clouds_all" value="20" min="0" max="100">
    </label>
    <label>Тип погоды
      <select id="weather_main">
        <option>Clear</option><option>Clouds</option><option>Rain</option>
        <option>Snow</option><option>Fog</option><option>Haze</option>
        <option>Mist</option><option>Drizzle</option><option>Thunderstorm</option>
        <option>Squall</option><option>Smoke</option>
      </select>
    </label>
    <label>Описание погоды
      <input type="text" id="weather_description" value="sky is clear">
    </label>
    <button onclick="predict()">Предсказать трафик</button>
    <div id="result"></div>
  </div>

  <script>
    async function predict() {
      const dt = document.getElementById('date_time').value.replace('T', ' ') + ':00';
      const body = {
        date_time:           dt,
        temp:                parseFloat(document.getElementById('temp').value),
        rain_1h:             parseFloat(document.getElementById('rain_1h').value),
        snow_1h:             parseFloat(document.getElementById('snow_1h').value),
        clouds_all:          parseInt(document.getElementById('clouds_all').value),
        weather_main:        document.getElementById('weather_main').value,
        weather_description: document.getElementById('weather_description').value,
      };
      const res = document.getElementById('result');
      try {
        const r = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || r.statusText);
        res.className = 'ok';
        res.textContent = 'Прогноз: ' + Math.round(data.traffic_volume) + ' машин/час';
      } catch(e) {
        res.className = 'err';
        res.textContent = 'Ошибка: ' + e.message;
      }
      res.style.display = 'block';
    }
  </script>
</body>
</html>"""


def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found. Run `python -m src.train` first.")
        _model = load_model(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    date_time: str
    temp: float
    rain_1h: float = 0.0
    snow_1h: float = 0.0
    clouds_all: int = 0
    holiday: str = "None"
    weather_main: str = "Clear"
    weather_description: str = "sky is clear"


class PredictResponse(BaseModel):
    traffic_volume: float


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _get_model()
    row = pd.DataFrame([req.model_dump()])
    row["date_time"] = pd.to_datetime(row["date_time"])
    row = add_time_features(row)
    X = row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    pred = model.predict(X)
    return PredictResponse(traffic_volume=float(pred[0]))
