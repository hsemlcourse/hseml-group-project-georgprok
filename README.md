# ML Project — Прогнозирование автомобильного трафика

**Студент:** Прокофьев Георгий Александрович

**Группа:** БИВ231

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

**Задача:** регрессия.

**Датасет:** Metro Interstate Traffic Volume.

- UCI: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- Kaggle: https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume

**Целевая переменная:** `traffic_volume`.

**Целевая метрика:** RMSE. Дополнительно считаются MAE и R2.

Цель проекта — предсказать почасовой объём автомобильного трафика по временным признакам и погодным условиям.

## Структура репозитория

```text
.
├── data
│   ├── Metro_Interstate_Traffic_Volume.csv  # Исходный датасет
│   └── processed                            # Очищенные и обработанные данные
├── models
│   ├── best_model.joblib                    # Сохранённая лучшая модель
│   └── metrics.json                         # Метрики всех моделей
├── notebooks
│   ├── 01_eda.ipynb                         # EDA и feature engineering
│   ├── 02_baseline.ipynb                    # Baseline-модели (CP1-2)
│   └── 03_experiments.ipynb                 # Тюнинг, ablation, feature importance (CP3)
├── presentation                             # Презентация для защиты
├── report
│   └── report.md                            # Отчёт
├── src
│   ├── __init__.py
│   ├── preprocessing.py                     # Загрузка, очистка, feature engineering, split, preprocessor
│   ├── modeling.py                          # Модели, метрики, гиперпараметры, save/load
│   ├── train.py                             # End-to-end обучение (python -m src.train)
│   └── service.py                           # FastAPI-сервис инференса
├── tests
│   └── test.py                              # 18 pytest-тестов на src/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                           # Ruff + pytest config
├── requirements.txt
└── README.md
```

## Запуск

### 1. Установить зависимости

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Обучить модели

```bash
python -m src.train
```

Скрипт обучит 8 моделей, затем тюнит лучшие через `RandomizedSearchCV` + `TimeSeriesSplit`, выберет лучшую по validation RMSE, переобучит на train+valid и сохранит:
- `models/best_model.joblib`
- `models/metrics.json`

### 3. Запустить сервис

```bash
uvicorn src.service:app --host 0.0.0.0 --port 8000
```

Эндпоинты:
- `GET /health` — проверка работоспособности
- `POST /predict` — прогноз `traffic_volume`

Пример запроса:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"date_time": "2018-06-15 08:00:00", "temp": 295.0, "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 20, "weather_main": "Clear", "weather_description": "sky is clear", "holiday": "None"}'
```

### 4. Запустить через Docker

```bash
docker-compose up --build
```

Сервис поднимется на `http://localhost:8000`.

### 5. Тесты

```bash
pytest tests/ -v
```

### 6. Линтер

```bash
ruff check src/ --line-length 120
```

### 7. Jupyter-ноутбуки

```bash
jupyter notebook
```

Открыть в порядке:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_baseline.ipynb`
3. `notebooks/03_experiments.ipynb`

## Данные

- UCI: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- Kaggle: https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume
- Локально: `data/Metro_Interstate_Traffic_Volume.csv`

48 204 почасовых наблюдения трафика на I-94 Minneapolis/St Paul (2012–2018). Признаки: дата и время, погода, праздники. Целевая переменная: `traffic_volume`.

## Результаты

Временной сплит (без data leakage):

- train: `date_time < 2017-01-01` (29 643 строки)
- validation: `2017-01-01 – 2017-12-31` (10 596 строк)
- test: `date_time >= 2018-01-01` (7 948 строк)

### Baseline (8 моделей, validation RMSE)

| Модель | RMSE | MAE | R2 |
|---|---:|---:|---:|
| HistGradientBoosting | 491.71 | 315.11 | 0.9387 |
| RandomForest | 495.90 | 314.14 | 0.9377 |
| ExtraTrees | 543.61 | 336.12 | 0.9251 |
| DecisionTree | 690.25 | 404.29 | 0.8793 |
| KNN | 969.75 | 678.39 | 0.7617 |
| Lasso | 1574.55 | 1348.06 | 0.3718 |
| Ridge | 1574.71 | 1347.07 | 0.3717 |
| Dummy mean | 1989.58 | 1748.43 | -0.003 |

### После тюнинга (RandomizedSearchCV + TimeSeriesSplit)

| Модель | Validation RMSE | Лучшие параметры |
|---|---:|---|
| **HistGradientBoosting (tuned)** | **473.69** | max_iter=100, max_depth=5, lr=0.05, min_samples_leaf=20 |
| RandomForest (tuned) | 619.94 | n_estimators=200, max_depth=20, min_samples_leaf=1 |
| ExtraTrees (tuned) | 822.68 | n_estimators=200, max_depth=20, min_samples_leaf=1 |

### Финальный тест

| Модель | RMSE | MAE | R2 |
|---|---:|---:|---:|
| HistGradientBoosting (tuned) | **503.68** | **288.97** | **0.9348** |

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
