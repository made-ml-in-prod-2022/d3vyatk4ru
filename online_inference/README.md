# d3vyatk4ru
Делаем online inference для модели машинного обучения

#### Запуск тестов (Надо быть в директории online inference)
Для Windows:
~~~
python -m pytest tests\tests.py
~~~

Для Linux:
~~~
pytest tests/tests.py
~~~

## Запуск сервиса в контейнере

#### Загрузка образа из ретозитория hub.docker
~~~
docker pull d3vyatk4ru/d3onin:latest
~~~

#### Запуск контейнера с docker образом
~~~
docker run --rm -p 9090:9090 d3vyatk4ru/d3onin:latest
~~~

## Оптимизация docker образа
Сначала использовался образ Ubuntu, поэтому места занимало очень много. Далее переехал на образ Python:3.8.10 --
получилось около 1,5GB. Далее прочитал про python:3.8.10-slim. Места было уже ~700MB. Далее избавился от ненужных
пакетов в requirements.txt, и образ стал весить 594MB. Так же было уменьшено количество слоев и копируемых файлов.
И еще использовался параметр --no-cache-dir для установки зависимостей.