## Как запустить обучение

В файле конфигурации `config/paths.json` необходимо указать
локальные пути:
- `dataset_dir`: путь до директории с датасетом TinyStories.
Если в ней есть json-файлы, то будут использованы они.
Если нет, то датасет будет загружен автоматически.
В эту же директорию будут записаны токенизированные тексты.
- `data_writeable_dir`: в случае, если директория с датасетом доступна
только для чтения, следует указать путь до директории с правами на запись.
Если не указывать, то будет совпадать с `dataset_dir`.

Использование обоих аргументов может быть полезно при работе с Kaggle,
где датасеты смонтированы в директорию, доступную только для чтения.

---

В файле конфигурации `config/logger.json` необходимо указать:
- `project_name`: имя проекта в WANDB
- `token_env_var_name`: имя переменной окружения, в которой хранится токен WandB

---

Далее обучение запускается командой
```commandline
python3 train.py
```

Пример обучения модели и её выходы - в `notebooks/demo_training.ipynb`.
