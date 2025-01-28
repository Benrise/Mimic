import logging

from typing import List
from openai import OpenAI
from httpx import Client
from uuid import UUID

from .database import select_messages_by_dialog
from .schemas import GetMessageRequestModel
from .config import PROXY_URL, OPEN_AI_API_KEY, SYS_PROMPT

logger = logging.getLogger(__name__)

# Инициализируем OpenAI-клиент
if OPEN_AI_API_KEY and PROXY_URL:
    client = OpenAI(
        api_key=OPEN_AI_API_KEY,
        http_client=Client(proxy=PROXY_URL)
    )


def build_openai_messages(dialog_id: UUID, last_msg_text: str) -> List[dict]:
    """
    Собирает весь контекст диалога из БД, преобразует
    в формат сообщений для ChatCompletion (role: user/assistant).
    Добавляет текущее новое сообщение пользователя в конце.
    """
    # participant_index=0 => user, participant_index=1 => assistant
    db_messages = select_messages_by_dialog(dialog_id)

    messages_for_openai = []
    messages_for_openai.append({"role": "system", "content": SYS_PROMPT})

    for msg in db_messages:
        role = "user" if msg["participant_index"] == 0 else "assistant"
        messages_for_openai.append({"role": role, "content": msg["text"]})

    messages_for_openai.append({"role": "user", "content": last_msg_text})
    return messages_for_openai


def query_openai_with_context(body: GetMessageRequestModel, model: str = "gpt-4o") -> str:
    """
    Формирует сообщения для OpenAI, включая весь контекст диалога,
    затем отправляет запрос и возвращает текст ответа.
    """
    logger.info(f"Using model: {model}")

    messages = build_openai_messages(body.dialog_id, body.last_msg_text)

    # Делаем запрос к OpenAI ChatCompletion
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    logger.info(str(chat_completion))

    answer_text = chat_completion.choices[0].message.content
    logger.info(f"OpenAI answer: {answer_text}")
    return answer_text


def query_openai_without_context(prompt: str, model: str = "gpt-4o") -> str:
    """
    Формирует сообщения для OpenAI, без контекста диалога,
    затем отправляет запрос и возвращает текст ответа.
    """
    logger.info(f"Using model: {model}")
    
    messages_for_openai = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt}
    ]

    # Делаем запрос к OpenAI ChatCompletion
    chat_completion = client.chat.completions.create(
        messages=messages_for_openai,
        model=model,
    )
    logger.info(str(chat_completion))

    answer_text = chat_completion.choices[0].message.content
    logger.info(f"OpenAI answer: {answer_text}")
    return answer_text
