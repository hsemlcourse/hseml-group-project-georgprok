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

**Датасет:** Metro Interstate Traffic Volume, известный по UCI/Kaggle.

**Целевая переменная:** `traffic_volume`.

**Целевая метрика:** RMSE. Дополнительно считаются MAE и R2.

Цель проекта - предсказать почасовой объем автомобильного трафика по временным признакам и погодным условиям.

## Структура репозитория

```text
.
├── data
│   ├── Metro_Interstate_Traffic_Volume.csv  # Исходный датасет
│   ├── processed                            # Очищенные и обработанные данные
│   └── raw                                  # Исходные файлы
├── models                                   # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb                         # EDA
│   └── 02_baseline.ipynb                    # Baseline и первые модели
├── presentation                             # Презентация для защиты
├── report
│   └── report.md                            # Отчёт
├── src
├── tests
│   └── test.py                              # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить Jupyter
jupyter notebook
```

После запуска открыть ноутбуки:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_baseline.ipynb`

## Данные

- `data/Metro_Interstate_Traffic_Volume.csv` — исходный датасет.
- `data/raw/` — папка для исходных файлов.
- `data/processed/` — предобработанные данные.

В данных есть дата и время, погодные признаки, информация о празднике и целевая переменная `traffic_volume`.

## Результаты

В CP1 использован временной split, без random split:

- train: `date_time < 2017-01-01`
- validation: `2017-01-01 <= date_time < 2018-01-01`
- test: `date_time >= 2018-01-01`

| Модель | RMSE | MAE | Примечание |
|---|---:|---:|---|
| Dummy mean | 1989.58 | 1748.43 | Наивный baseline |
| Ridge Regression | 1574.71 | 1347.07 | Линейная модель |
| Random Forest | 495.90 | 314.14 | Почти как лучшая модель |
| HistGradientBoosting | 491.71 | 315.11 | Лучшая модель по validation RMSE |

На test выбранная модель `HistGradientBoosting` получила RMSE `472.28`, MAE `270.89`, R2 `0.9427`.

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
