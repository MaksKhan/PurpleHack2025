# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем необходимые зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir numpy && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Копируем файлы приложения в контейнер
COPY scheduler.py /app/scheduler.py

# Делаем скрипт исполняемым
RUN chmod +x /app/scheduler.py

# Запускаем CLI-приложение, которое будет читать JSON из stdin и писать в stdout
ENTRYPOINT ["python", "/app/scheduler.py"]