import os

DB_USER = os.getenv("DB_USER", "student")
DB_PASSWORD = os.getenv("DB_PASSWORD", "student_pass")
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "chat_db")

# Формируем строку подключения БД
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# API ChatGPT
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "")
PROXY_URL = os.getenv("PROXY_URL", "")
API_PORT = os.getenv("API_PORT", "8000")

# Промпт, добавляемый перед пользовательским текстом:
SYS_PROMPT = os.getenv("SYS_PROMPT", "")
