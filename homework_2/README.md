# HW2 — воспроизведение «Super-Convergence» (Smith & Topin, 2017)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alberelloo/Efficient-Models-course-ITMO-2026/blob/main/homework_2/super_convergence.ipynb)

Воспроизведение статьи [arXiv:1708.07120](https://arxiv.org/abs/1708.07120) на CIFAR-10 с ResNet-56 на одном GPU Google Colab: LR range test, сравнение политики **1cycle** со стандартным ступенчатым расписанием и оценка оптимального LR упрощённым Hessian-free методом.

📄 **Полный отчёт: [REPORT.md](REPORT.md)**

## Итоги

| Утверждение | Вердикт |
|---|---|
| C1. Тест диапазона LR: точность ResNet-56 не падает при росте LR до 3 (нет «пика и спада») | 🟡 частично |
| C2. 1cycle с большим LR достигает той же или лучшей точности за кратно меньшее число итераций | ✅ подтверждается |
| C3. Оценка оптимального LR (ур. 8) при 1cycle остаётся высокой, при стандартном режиме падает | ✅ подтверждается |
| C4. Выигрыш super-convergence растёт при уменьшении объёма обучающих данных | ⏸ не запускалось |

| Прогон | Расписание | Эпох | Final test acc | Best test acc | Время, мин | Картинок/с | Расходимость | Пропущено шагов fp16 |
|---|---|---|---|---|---|---|---|---|
| lr_range_test | Тест диапазона LR, 5 000 итераций | 100 | 58.47% | 82.73% | 16 | 5 388 | нет | 0 |
| one_cycle | 1cycle, 4 000 итераций | 80 | 93.03% | 93.11% | 13 | 5 324 | нет | 0 |
| piecewise_equal_budget | Ступенчатый LR, 4 000 итераций | 80 | 82.25% | 82.47% | 13 | 5 416 | нет | 0 |
| piecewise_long_budget | Ступенчатый LR, 12 000 итераций | 240 | 87.68% | 87.78% | 38 | 5 313 | нет | 0 |

## Структура

```
homework_2/
├── README.md                  этот файл
├── REPORT.md                  отчёт о воспроизведении (цель, метод, установка, результаты, отклонения, выводы)
├── super_convergence.ipynb    ноутбук Colab: код экспериментов + выводы ячеек
├── figures/                   графики отчёта (PNG)
│   ├── 01_lr_range_test.png
│   ├── 02_one_cycle_vs_piecewise.png
│   ├── 03_train_vs_test.png
│   └── 04_hessian_free_lr.png
└── results/
    ├── <прогон>.json          кривые обучения и метрики каждого прогона (+ конфигурация и её хэш)
    ├── summary.csv            сводная таблица прогонов
    └── environment.json       GPU, версии библиотек, дата
```

## Как запустить

1. Открыть ноутбук по кнопке «Open in Colab» выше.
2. `Runtime → Change runtime type → T4 GPU`.
3. Добавить секрет `GITHUB_TOKEN_EFFICIENT_NN` (значок ключа слева) и включить для него «Notebook access». Нужен токен с правом записи в репозиторий: fine-grained PAT с `Contents: Read and write` или classic PAT со scope `repo`.
4. `Runtime → Run all`. Ноутбук клонирует репозиторий, коммитит результаты после каждого прогона и в конце пушит отчёт, README и исполненную копию ноутбука.

Весь план — около 500 эпох CIFAR-10; точный прогноз времени печатается перед обучением. Если сессия оборвётся, достаточно снова выполнить `Run all`: готовые прогоны подтянутся из `results/` и не будут пересчитываться.

Без GitHub: `PUSH_TO_GITHUB = False` в ячейке конфигурации — всё сохранится в `/content/homework_2`.
