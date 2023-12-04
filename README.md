## Preliminaries

```commandline
pip install -r requirements.txt
```

## Как запустить обучение

В файле конфигурации `config/paths.json` необходимо указать
локальные пути:
- `dataset_dir`: путь до директории с датасетом TinyStories.
Если в ней есть json-файлы, то будут использованы они.
Если нет, то датасет будет загружен автоматически.
В эту же директорию будут записаны токенизированные тексты.

---

В файле конфигурации `config/logger.json` необходимо указать:
- `project_name`: имя проекта в WANDB
- `token_env_var_name`: имя переменной окружения, в которой хранится токен WandB

---

Гиперпараметры модели указываются в файле конфигурации `config/model.json`.
Примеры конфигураций - в директории `model_configs`.

Чтобы не ожидать обработки и токенизации всего датасета (занимает около получаса), можно загрузить уже
токенизированный датасет: [файл1](https://drive.google.com/file/d/1GkZ7PR6F-mR-H9g4GVi9D5zXFXbYNy0a/view?usp=drive_link), [файл2](https://drive.google.com/file/d/1vTK0_R7xvTBzVPzfbI1uS5gBCXV_u6kL/view?usp=drive_link),
оба файла необходимо разместить в директории, указанной в качестве `dataset_dir` в конфиге `config.paths.json`.

Далее обучение запускается командой
```commandline
python3 train.py
```

## Как запустить inference обученной модели

Достаточно запустить
```commandline
python3 inference.py
```

Будет загружен чекпоинт модели и обученный энкодер.
Если по каким-то причинам автоматическое скачивание не работает, то
- [чекпоинт](https://drive.google.com/file/d/1Fsknyk8DWbI1nASTISscU2Kwxbr3gKYl/view?usp=sharing),
его поместить по пути `inference/checkpoint.pth`
- [энкодер](https://drive.google.com/file/d/1vOVg500jLVlqTKhXeeNeL2H5DTtn_Wiv/view?usp=drive_link),
содержимое архива разместить в директории `inference/encoder`

После скачивания весов и энкодера, в терминале будет ожидаться prompt для генерации.
