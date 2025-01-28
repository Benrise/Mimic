import time
import uvicorn
import logging
import psycopg2

from uuid import uuid4
from fastapi import FastAPI

from src import database
from src.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, PROXY_URL, OPEN_AI_API_KEY, API_PORT
from src.schemas import GetMessageResponseModel, GetMessageRequestModel
from src.gpt_api import query_openai_with_context

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GPT Bot Service",
    description="Сервис для генерации ответов",
)


@app.on_event("startup")
def on_startup() -> None:
    """
    Запуск приложения FastAPI.
    Выполняем проверку доступности PostgreSQL в цикле (на всякий случай)
    После успешного соединения инициализируем базу.
    """
    while True:
        try:
            conn = psycopg2.connect(
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            conn.close()
            break
        except psycopg2.OperationalError:
            logger.warning("Waiting for PostgreSQL to become available...")
            time.sleep(2)

    # Инициализация БД
    database.init_db()


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel) -> GetMessageResponseModel:
    """
    Эндпоинт, принимающий сообщение от пользователя и возвращающий ответ GPT.

    Действия:
    1. Сохраняет входное сообщение (participant_index=0) в БД.
    2. Загружает весь контекст диалога (user + assistant) и формирует запрос к GPT.
    3. Генерирует ответ OpenAI ChatCompletion.
    4. Сохраняет ответ бота (participant_index=1) в БД.
    5. Возвращает ответ и dialog_id.
    """
    # Сохраняем новое пользовательское сообщение
    user_msg_id = body.last_message_id or uuid4()
    database.insert_message(
        msg_id=user_msg_id,
        dialog_id=body.dialog_id,
        text=body.last_msg_text,
        participant_index=0
    )

    response_from_openai = "Service unavailable"
    # Генерируем ответ GPT
    if OPEN_AI_API_KEY and PROXY_URL:
        response_from_openai = query_openai_with_context(body, model="gpt-4o")

    # Сохраняем сообщение бота
    bot_msg_id = uuid4()
    database.insert_message(
        msg_id=bot_msg_id,
        dialog_id=body.dialog_id,
        text=response_from_openai,
        participant_index=1
    )

    return GetMessageResponseModel(
        new_msg_text=response_from_openai,
        dialog_id=body.dialog_id
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
