# Отчет по проекту

Студент: Прокофьев Георгий Александрович Группа: 231

## 1. Постановка задачи

Цель проекта — прогнозировать почасовой объем автомобильного трафика `traffic_volume` на основе времени и погодных условий.

Это задача регрессии. Основная метрика — RMSE, потому что она сильнее штрафует большие ошибки. Дополнительно считаются MAE и R2. MAE удобно читать как среднюю абсолютную ошибку прогноза в машинах в час.

## 2. Описание данных

Датасет: Metro Interstate Traffic Volume.

- UCI: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- Kaggle: https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume

Файл в проекте: `data/Metro_Interstate_Traffic_Volume.csv`. В нем есть почасовые наблюдения трафика на I-94 около Minneapolis/St Paul, признаки погоды и информация о праздниках.

Исходный размер: 48204 строки и 9 колонок.

Колонки:
- `holiday` — праздник или обычный день;
- `temp`, `rain_1h`, `snow_1h`, `clouds_all` — погодные признаки;
- `weather_main`, `weather_description` — тип погоды;
- `date_time` — время наблюдения;
- `traffic_volume` — целевая переменная.

## 3. Обработка данных

Что сделано в `src/preprocessing.py` (логика та же, что в `notebooks/01_eda.ipynb`):

- `date_time` приведен к datetime;
- данные отсортированы по времени;
- 48143 пропуска в `holiday` заменены на `None`;
- удалены 17 полных дубликатов;
- 10 значений `temp <= 1 K` помечены как пропуски;
- добавлены признаки `hour`, `day_of_week`, `month`, `year`, `is_weekend`, `is_rush_hour`.

После feature engineering получилось 48187 строк и 15 колонок.

Для моделей пропуски в числовых признаках заполняются медианой внутри `Pipeline`, обученного только на train.

## 4. EDA

Построены графики в `notebooks/01_eda.ipynb`:

- распределение `traffic_volume`;
- динамика `traffic_volume` по времени;
- средний трафик по часу, дню недели и месяцу;
- `traffic_volume` по `weather_main`;
- связь `temp` и `traffic_volume`;
- correlation heatmap.

Главный вывод: трафик сильно зависит от времени суток и дня недели. В будние дни и часы поездок средний трафик выше. Погодные признаки тоже помогают, но сами по себе не объясняют весь разброс.

## 5. Train / validation / test split

Используется временной split:

- train: `date_time < 2017-01-01`, 29643 строки;
- validation: `2017-01-01 <= date_time < 2018-01-01`, 10596 строк;
- test: `date_time >= 2018-01-01`, 7948 строк.

Random split не используется: для временных данных он приводит к data leakage (модель видит будущие периоды при обучении).

## 6. Baseline и первые модели (CP1-2)

Все модели обучены в `Pipeline` с `ColumnTransformer`.

Числовые признаки: `SimpleImputer(strategy="median")` + `StandardScaler`.

Категориальные признаки: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.

`RANDOM_STATE = 42`.

Validation results (4 модели из CP1-2):

| model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| HistGradientBoosting | 491.71 | 315.11 | 0.9387 |
| Random Forest | 495.90 | 314.14 | 0.9377 |
| Ridge Regression | 1574.71 | 1347.07 | 0.3717 |
| Dummy mean | 1989.58 | 1748.43 | -0.0030 |

## 7. CP3: расширение и тюнинг гиперпараметров

### 7.1. Структура src/

Весь переиспользуемый код вынесен из ноутбуков в модули:

- `src/preprocessing.py` — загрузка, очистка, feature engineering, временной сплит, ColumnTransformer;
- `src/modeling.py` — 8 моделей-кандидатов, метрики, сетки гиперпараметров, save/load модели;
- `src/train.py` — end-to-end обучение: запускается как `python -m src.train`;
- `src/service.py` — FastAPI-сервис инференса.

Ноутбук `notebooks/03_experiments.ipynb` импортирует функции из `src/` и не дублирует логику.

### 7.2. Расширенный список моделей (8 штук)

Validation results (все 8 baseline-моделей):

| model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| HistGradientBoosting | 491.71 | 315.11 | 0.9387 |
| RandomForest | 495.90 | 314.14 | 0.9377 |
| ExtraTrees | 543.61 | 336.12 | 0.9251 |
| DecisionTree | 690.25 | 404.29 | 0.8793 |
| KNN | 969.75 | 678.39 | 0.7617 |
| Lasso | 1574.55 | 1348.06 | 0.3718 |
| Ridge | 1574.71 | 1347.07 | 0.3717 |
| Dummy mean | 1989.58 | 1748.43 | -0.003 |

### 7.3. Тюнинг гиперпараметров

Тюнинг проводится через `RandomizedSearchCV` с `TimeSeriesSplit(n_splits=3)` на тренировочных данных. `TimeSeriesSplit` создаёт фолды в хронологическом порядке, что исключает data leakage (модель не видит будущее при кросс-валидации).

Тюнились три лучшие tree-based модели: `RandomForest`, `ExtraTrees`, `HistGradientBoosting`. Параметры: `n_iter=10`, `scoring="neg_root_mean_squared_error"`.

**Сетки гиперпараметров:**

HistGradientBoosting:
- `max_iter`: [100, 200, 300]
- `learning_rate`: [0.05, 0.08, 0.1, 0.15]
- `max_depth`: [None, 5, 8]
- `l2_regularization`: [0.0, 0.1, 1.0]
- `min_samples_leaf`: [20, 40, 60]

RandomForest / ExtraTrees:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, None]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ["sqrt", "log2"]

**Результаты тюнинга (validation RMSE):**

| model | RMSE | MAE | R2 | Лучшие параметры |
|---|---:|---:|---:|---|
| HistGradientBoosting (tuned) | **473.69** | 303.70 | 0.9431 | max_iter=100, max_depth=5, lr=0.05, min_samples_leaf=20 |
| RandomForest (tuned) | 619.94 | 460.07 | 0.9026 | n_estimators=200, max_depth=20, min_samples_leaf=1 |
| ExtraTrees (tuned) | 822.68 | 657.00 | 0.8285 | n_estimators=200, max_depth=20, min_samples_leaf=1 |

Лучшая модель по validation RMSE — `HistGradientBoosting (tuned)`. Тюнинг улучшил RMSE с 491.71 до 473.69 (−3.7%).

Примечание: для RandomForest и ExtraTrees тюнинг на временных фолдах train-множества подобрал параметры, которые лучше работают внутри обучающего периода (2012–2016), но не дали улучшения на validation 2017. Это особенность temporal data — распределение 2017 отличается от внутренних cv-фолдов.

### 7.4. Ablation study

Сравнивали `HistGradientBoosting (tuned)` на двух наборах признаков:

| Конфигурация | Признаки | Validation RMSE |
|---|---|---:|
| time_only | hour, day_of_week, month, year, is_weekend, is_rush_hour | см. 03_experiments.ipynb |
| time_weather | все признаки (время + погода) | 473.69 |

Добавление погодных признаков снижает RMSE. Временные признаки — основной драйвер качества, что согласуется с EDA.

### 7.5. Финальные метрики на test (одно измерение)

Лучшая модель переобучена на train+valid и оценена на test один раз:

| model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| HistGradientBoosting (tuned) | **503.68** | **288.97** | **0.9348** |

### 7.6. Сервис инференса (FastAPI)

`src/service.py` реализует REST API:

- `GET /health` — возвращает `{"status": "ok"}`;
- `POST /predict` — принимает сырое наблюдение (date_time, temp, погода, holiday), вычисляет временные признаки через `add_time_features`, возвращает прогноз `traffic_volume`.

Запуск: `uvicorn src.service:app --host 0.0.0.0 --port 8000`

### 7.7. Docker

`Dockerfile` и `docker-compose.yml` упаковывают сервис:

```
docker-compose up --build
```

Сервис доступен на `http://localhost:8000`.

### 7.8. Линтер и тесты

- `ruff check src/ --line-length 120` — **0 ошибок**. Конфиг в `pyproject.toml`.
- `pytest tests/ -v` — **18/18 тестов прошли**.

Тесты покрывают: очистку данных, feature engineering, отсутствие пересечений в сплите, изоляцию таргета от признаков, препроцессор, метрики, обучение pipeline.

CI (`.github/workflows/ci.yml`) запускает `ruff` и `pytest` при каждом push.

## 8. Краткие выводы

Baseline (Dummy) сильно уступает всем обученным моделям. Линейные модели (Ridge, Lasso) заметно лучше baseline, но деревья и бустинг лучше ловят нелинейность. Добавление тюнинга улучшило лучшую модель с RMSE 491.71 до 473.69 на validation. Финальная оценка на test: RMSE=503.68, MAE=288.97, R2=0.9348. Test использовался строго один раз после выбора модели по validation.
