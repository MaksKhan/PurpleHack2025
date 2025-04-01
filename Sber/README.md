# RL-алгоритм планирования задач

```bash
docker build -t project-optimizer .
docker run -p 8000:8000 project-optimizer
```
## Использование:
При запуске разворачивается FastAPI-сервис с эндпоинтом predict. Можно его посмотреть по localhost:8000/docs.
На вход: {project_data: JSON}