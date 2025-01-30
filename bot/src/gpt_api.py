import logging
import random
import time

from typing import List
from openai import OpenAI
from httpx import Client
from uuid import UUID

from .utils import humanSleep
from .database import select_messages_by_dialog
from .schemas import GetMessageRequestModel
from .config import PROXY_URL, OPEN_AI_API_KEY, SYS_PROMPT

logger = logging.getLogger(__name__)

# Функции постобработки

def random_typo(text):
    replacements = {
        'а': 'ф', 'б': 'и', 'в': 'а', 'г': 'п', 'д': 'л',
        'е': 'у', 'ё': 'е', 'ж': 'о', 'з': 'х', 'и': 'м',
        'й': 'ц', 'к': 'р', 'л': 'д', 'м': 'и', 'н': 'ы',
        'о': 'щ', 'п': 'г', 'р': 'к', 'с': 'ы', 'т': 'и',
        'у': 'е', 'ф': 'а', 'х': 'з', 'ц': 'й', 'ч': 'с',
        'ш': 'щ', 'щ': 'ш', 'ы': 'н', 'ь': 'т', 'э': 'е',
        'ю': 'ж', 'я': 'ч'
    }
    
    methods = ['replacement', 'deletion', 'insertion', 'transposition']
    method = random.choice(methods)
    
    if len(text) == 0:
        return text
    
    if method == 'replacement':
        index = random.randint(0, len(text) - 1)
        if text[index] in replacements:
            typo_char = replacements[text[index]]
            text = text[:index] + typo_char + text[index + 1:]
    elif method == 'deletion':
        if len(text) > 1:
            index = random.randint(0, len(text) - 1)
            text = text[:index] + text[index + 1:]
    elif method == 'insertion':
        index = random.randint(0, len(text) - 1)
        char_to_insert = random.choice(list(replacements.keys()))
        text = text[:index] + char_to_insert + text[index:]
    elif method == 'transposition':
        if len(text) > 1:
            index = random.randint(0, len(text) - 2)
            text = text[:index] + text[index + 1] + text[index] + text[index + 2:]

    return text

def introduce_typos(text, typos=1):
    length = len(text)
    num_typos = length * typos // 40
    
    words = text.split()
    
    for _ in range(num_typos):
        word_index = random.randint(0, len(words) - 1)
        words[word_index] = random_typo(words[word_index])
    
    return ' '.join(words)

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
    
    humanSleep(answer_text)
        
    return answer_text


def query_openai_with_local_context(dialog: list, model: str = "gpt-4o") -> str:
    """
    Формирует сообщения для OpenAI, с локальным контекстом диалога,
    затем отправляет запрос и возвращает текст ответа.
    """
    logger.info(f"Using model: {model}")
    
    # Делаем запрос к OpenAI ChatCompletion
    chat_completion = client.chat.completions.create(
        messages=dialog,
        model=model,
    )
    
    logger.info(str(chat_completion))
    
    answer_text = introduce_typos(chat_completion.choices[0].message.content)
    
    logger.info(f"OpenAI answer: {answer_text}")
    
    humanSleep(answer_text)
    
    return answer_text


