import time
import uvicorn
import logging
import random
import psycopg2

from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src import database
from src.config import (
    DB_USER, 
    DB_PASSWORD, 
    DB_HOST, 
    DB_PORT, 
    DB_NAME, 
    PROXY_URL, 
    OPEN_AI_API_KEY,
    API_PORT, 
    SYS_PROMPT, 
    BOT_NAMES,
    BOT_REGISTERS,
    BOT_AGES,
    BOT_SPECIALIZATION,
    BOT_FAVORITE_TRASH_WORDS,
    BOT_GREETINGS,
)
from src.schemas import GetMessageResponseModel, GetMessageRequestModel
from src.gpt_api import query_openai_with_context, query_openai_with_local_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
Имитация локальной БД для песочницы (/playground, /reset_playground, /get_playground_data)
"""
sys_prompt_filled = SYS_PROMPT.format(
    random_name=random.choice(BOT_NAMES),
    random_register=random.choice(BOT_REGISTERS),
    random_age=random.choice(BOT_AGES),
    random_specialization=random.choice(BOT_SPECIALIZATION),
    random_favorite_trash_word=random.choice(BOT_FAVORITE_TRASH_WORDS),
    random_greeting=random.choice(BOT_GREETINGS)
)
dialog_history = [{"role": "system", "content": sys_prompt_filled}]


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
        response_from_openai = query_openai_with_context(body)

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

@app.post("/playground", response_class=HTMLResponse)
async def playground(query: str):
    """
    Эндпоинт для тестирования взаимодействия с ботом в режиме реального времени c локальным сохранением.
    
    Действия:
    1. Принимает текстовый запрос от пользователя (query).
    2. Добавляет запрос пользователя в историю диалога (dialog_history) с ролью "user".
    3. Генерирует ответ от модели GPT, используя текущий контекст диалога (dialog_history).
    4. Добавляет ответ модели в историю диалога с ролью "assistant".
    5. Форматирует последние сообщения из истории диалога (исключая первый элемент) 
       для удобного вывода в HTML-формате.
    6. Возвращает отформатированный диалог в виде HTML-страницы.
    """
    global dialog_history
    dialog_history.append({"role": "user", "content": query})
    
    response_from_openai = "Service unavailable"
    
    if OPEN_AI_API_KEY and PROXY_URL:
        response_from_openai = query_openai_with_local_context(dialog_history)
        dialog_history.append({"role": "assistant", "content": response_from_openai})
       
    filtered_history = dialog_history[1:]
    
    formatted_dialog = "\n".join(
        f"[{i}][{'Пользователь' if msg['role'] == 'user' else 'Бот'}]: {msg['content']}"
        for i, msg in enumerate(filtered_history)
    )
        
    return f"{formatted_dialog}"

@app.delete("/reset_playground")
async def reset_playground():
    """
    Эндпоинт для обнуления локальной истории диалога.
    """
    global dialog_history
    dialog_history = [{"role": "system", "content": sys_prompt_filled}]
    return "История диалога успешно обнулена до системного промпта." 

@app.get("/get_playground_data")
async def get_playground_data() -> list:
    global dialog_history
    
    filtered_dialog_history = dialog_history[1:]
    
    return filtered_dialog_history


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, debug=True)
