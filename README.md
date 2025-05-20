# PurpleHack 2025

> **2 место, PurpleHack 2025** — за шесть дней мы закрыли пять независимых ML‑кейсов от Avito, Т‑Банка, Сбера, МТС и ArenaData. Репозиторий содержит ноутбуки и утилиты, которые воспроизводят лучшее решение команды «Мы МИСИС 177!!!».

---
## 0. Статья
Очень подробно мы описали наше решение в этой статье на Хабр: https://habr.com/ru/companies/alfa/articles/900824/

## 1. Быстрый старт

```bash
# клонируем репо
$ git clone https://github.com/MaksKhan/PurpleHack2025.git
$ cd PurpleHack2025

# создаём окружение
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt  # или poetry install

# по одному модулю
$ jupyter lab ArenaData/traffic_guard.ipynb
```
Также в каждой из папок есть описание запуска

> **Зависимости:** Python ≥ 3.11, Jupyter Lab, Docker ≥ 26, NVIDIA GPU + CUDA 12 (опционально — для Avito & T‑Bank).

---

## 2. Структура репозитория

```
PurpleHack2025/
├─ Avito/               # CNN + Mask‑RCNN для доминантного цвета товара
├─ T‑Bank/              # RAG‑агент (Gemma 3 4B @ Ollama + Faiss)
├─ Sber/                # RL‑планировщик календаря (PPO)
├─ MTS/                 # Бин‑пакинг ВМ (BestFit + GA)
├─ ArenaData/           # Heuristic spike‑detector (>6 пиков исходящего)
```

---

## 3. Ключевой функционал

| Кейс          | Цель                                            | 
| ------------- | ----------------------------------------------- |
| **Avito**     | Классификация доминантного цвета по фото товара |
| **T‑Bank**    | Chat‑Shop‑ассистент c RAG по 1 M+ позиций       |
| **Сбер**      | Автопланировщик задач в календаре               |
| **МТС**       | Размещение ВМ по хостам (бин‑пакинг)            |
| **ArenaData** | Детект аномального исходящего трафика           |

---

## 4. Стек и бейджи

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter‑Lab-notebooks-orange?logo=jupyter)](https://jupyter.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2‑cpu%2Fcuda‑12-lightgrey?logo=pytorch)](https://pytorch.org)
[![Faiss](https://img.shields.io/badge/Faiss-vector‑search-green)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma3‑4B-brightgreen)](https://ollama.com/library/gemma3)
[![Stable‑Baselines3](https://img.shields.io/badge/SB3‑Zoo-RL-red)](https://github.com/DLR-RM/rl-baselines3-zoo)
[![Docker](https://img.shields.io/badge/Docker-container-blue?logo=docker)](https://www.docker.com)

---

## 5. Установка (подробно)

1. **Python env**  — `make init` скачает модели, создаст `.env`.
2. **GPU support** — пропишите `TORCH_CUDA_ARCH_LIST` и проверьте `nvidia-smi`.
3. **Ollama + Gemma**

   ```bash
   curl https://ollama.ai/install.sh | sh
   ollama pull gemma3:4b
   ```
4. **Переменные окружения** — см. `sample.env`.

---


## 6. Команда

| Участник            | Роль                              |
| ------------------- | --------------------------------- | 
| **Карпов Назар**       | ArenaData ML | 
| **Смирнов Павел**      | CV / Avito color‑CNN              |
| **Душенев Даниил**    | GA / MTS VM placer и RL / Сбер calendar             |
| **Хандусь Максим**   | RAG / T‑Bank shop‑assistant       | 
| **Кузнецов Даниил** | RAG / T‑Bank shop‑assistant                |

---

## 7. Roadmap

* [ ] GPU‑parallel GA → 90 × ускорение.
* [ ] Gemma 7B MoE без потери latency.
* [ ] Synthetic‑data генерация для RL‑агента.

PR‑ы и issue‑запросы приветствуются!

---

## 8. Лицензия

[MIT](LICENSE) © 2025 «Мы МИСИС 177!!!»
