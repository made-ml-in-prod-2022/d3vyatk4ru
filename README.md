# d3vyatk4ru
Production ready проект для решения задачи классификации

Установка: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Использование логистической регрессии из папки ml_project:
~~~
python src/train_pipeline.py configs/train_config_log_reg.yaml
~~~
Использование случайного леса из папки ml_project:
~~~
python src/train_pipeline.py configs/train_config_random_forest.yaml
~~~

Тесты из папки ml_project:
~~~
pytest tests/tests.py
~~~

Организация проекта
------------

    ├── LICENSE
    ├── README.md               <- Правила Использования проекта.
    ├── data
    │   ├── predicted           <- Предстазанные метки для predict_pipeline.py
    │   └── raw                 <- Реальные данные.
    │
    ├── models                  <- Модели, трансформеры и метрики.
    │
    ├── notebooks               <- Jupyter notebooks с предварительным анализом данных.
    │
    ├── requirements.txt        <- Необходимые пакеты для запуска обучения и предсказния.
    │
    ├── setup.py                <- Возможность установки проекта через менеджер pip.
    │
    ├── src                     <- Код для запуска пайплана.
    │   ├── __init__.py         <- Делает src Python модулем.
    │   │
    │   ├── data                <- Работа с данными.
    │   │
    │   ├── entity              <- Структуры с параметрами для работы модели.
    │   │
    │   ├── features            <- Преобразование сырых данных к признакам дял модели.
    │   │
    │   ├── models              <- Тренировки модели и использование готовой модели.
    │   │
    │   ├── predict_pipeline.py <- Пайплайн для прогноза на данных
    │   │
    │   ├── train_pipeline.py   <- Пайплайн для тренировки модели
    │
    ├── tests                   <- Тесты

--------