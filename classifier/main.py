import time
import uuid
import logging

import uvicorn
from fastapi import FastAPI, HTTPException
import psycopg2

from src import database
from src.schemas import IncomingMessage, Prediction, IncomingDialog
from src.model_inference import classify_text, classify_dialog
from src.config import (
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, INFERENCE_PORT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Inference Service",
    description="Классификация диалогов"
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

    # Инициализация схемы/таблиц в базе
    database.init_db()

@app.post("/playground")
async def playground(dialog: IncomingDialog):
    is_bot_probability = await classify_dialog(dialog.messages)

    return {
        "is_bot_probability": is_bot_probability
    }

@app.post("/predict", response_model=Prediction)
async def predict(msg: IncomingMessage) -> Prediction:
    """
    Эндпоинт для сохранения сообщения и получения вероятности того,
    что в диалоге участвует бот.

    1. Сохраняем входное сообщение в таблицу `messages`.
    2. Забираем все сообщения данного `dialog_id`.
    3. Применяем zero-shot классификатор.
    4. Возвращаем объект `Prediction`.
    """

    database.insert_message(
        id=msg.id,
        text=msg.text,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index
    )

    # Загружаем весь диалог
    conversation_text = database.select_messages_by_dialog(msg.dialog_id)
    if not conversation_text:
        raise HTTPException(
            status_code=404,
            detail="No messages found for this dialog_id"
        )

    logger.info("Conversation_text:", conversation_text)
    
    is_bot_probability = await classify_text(conversation_text)
    prediction_id = uuid.uuid4()

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=INFERENCE_PORT, debug=True)
