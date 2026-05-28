# PE34 — Rock-Paper-Scissors Transfer Learning

Домашнее задание по теме дообучения нейросетей и transfer learning.

## Датасет

Использован собственный датасет Kaggle:

https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

Классы:

- `rock`
- `paper`
- `scissors`

## Что реализовано

- загрузка датасета через Kaggle API;
- автоматический поиск папки с классами;
- приведение данных к формату PyTorch `ImageFolder`;
- разбиение на `train / val / test`;
- аугментации изображений;
- общая функция визуализации батча изображений;
- общая функция `train_model()` для обучения моделей;
- графики `accuracy/loss`;
- confusion matrix и classification report;
- сравнение ResNet-18 с нуля и ResNet-18 через Transfer Learning;
- дополнительное fine-tuning дообучение последнего блока `layer4`.

## Архитектура проекта

```text
PE34-rockpaperscissors-transfer-learning/
  README.md
  fine_tuning_rps.py
  model_conclusions.md
  requirements.txt
```

## Как запустить

1. Открой Google Colab или локальную среду с GPU.
2. Установи зависимости:

```bash
pip install -r requirements.txt
```

3. Получи `kaggle.json` в Kaggle: `Account → Create New API Token`.
4. Запусти код из `fine_tuning_rps.py` или перенеси его в Colab notebook.
5. После выполнения сравни итоговую таблицу результатов: `scratch` vs `transfer/fine-tune`.

## Вывод

Transfer Learning должен показать более высокую точность при меньшем числе эпох, потому что модель уже содержит предобученный feature extractor. Модель с нуля стартует со случайных весов и требует больше данных, больше эпох и больше вычислительных ресурсов. Для датасета `rock / paper / scissors` предобученная ResNet-18 обычно быстрее выходит на высокую точность и лучше обобщает на validation/test split.
