import asyncio
import logging

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from openai import AsyncOpenAI
from httpx import AsyncClient

from .config import (
    GIGACHAT_API_KEY, 
    GIGACHAT_VALIDATION_MODEL, 
    OPEN_AI_API_KEY, 
    OPENAI_VALIDATION_MODEL,
    BOTHUB_API_KEY,
    BOTHUB_API_BASE_URL,
    PROXY_URL, 
    VALIDATE_SYS_PROMPT
)


class BotClassifier():
    def __init__(self):
        self.api_keys = {
            'openai': OPEN_AI_API_KEY,
            'gigachat': GIGACHAT_API_KEY,
            'bothub': BOTHUB_API_KEY,
        }
        self.logger = logging.getLogger(__name__)
        self.proxy_url = PROXY_URL
        self.validate_sys_prompt = VALIDATE_SYS_PROMPT
        self.weights = {
            'openai': 0.6,
            'gigachat': 0.4,
        }

    async def _validate_with_openai(self, dialog, participant_index):
        model = OPENAI_VALIDATION_MODEL
        sys_prompt_validate_filled = VALIDATE_SYS_PROMPT.format(
            participant_index=participant_index
        )
        base_url = None
        
        if BOTHUB_API_BASE_URL or len(BOTHUB_API_BASE_URL) > 0:
            base_url = BOTHUB_API_BASE_URL
 
        chat = [{"role": "system", "content": sys_prompt_validate_filled}]
        chat.extend([{"role": "user", "content": line} for line in dialog])

        try:
            client = AsyncOpenAI(
                api_key=self.api_keys['openai'], 
                base_url=base_url, 
                http_client=AsyncClient(proxy=self.proxy_url)
            )
            response = await client.chat.completions.create(model=model,messages=chat)
            verdict = response.choices[0].message.content.lower()
            self.logger.info(f"OpenAI verdict: {verdict}")
            return float(verdict)

        except Exception as e:
            self.logger.warning(f"Error fetching from OpenAI: {e}")
            return 0.5

    async def _validate_with_gigachat(self, dialog, participant_index):
        model = GIGACHAT_VALIDATION_MODEL
        sys_prompt_validate_filled = VALIDATE_SYS_PROMPT.format(
            participant_index=participant_index
        )
        
        chat = Chat(messages=[Messages(role=MessagesRole.SYSTEM, content=sys_prompt_validate_filled)])
        chat.messages.extend([Messages(role=MessagesRole.USER, content=line) for line in dialog])
        
        try:
            async with GigaChat(
                    credentials=self.api_keys['gigachat'],
                    model=model,
                    verify_ssl_certs=False
                ) as giga:
                response = await asyncio.to_thread(lambda: giga.chat(chat))
                verdict = response.choices[0].message.content.lower()
                self.logger.info(f"GigaChat verdict: {verdict}")
                return float(verdict)
            
        except Exception as e:
            self.logger.warning(f"Error fetching from GigaChat: {e}")
            return 0.5
        
    def _extract_dialog_messages(self, dialog):
        """
        Разделяем диалог на два списка участников
        """
        dialog_lines = dialog.strip().splitlines()
        
        participant1_messages = []
        participant2_messages = []
        
        for line in dialog_lines:
            line = line.strip()
            
            if not line or len(line) < 3:
                        continue
            
            if line[0] == "0":
                participant1_messages.append(line[2:].strip())
            elif line[0] == "1":
                participant2_messages.append(line[2:].strip())
        
        return participant1_messages, participant2_messages

    async def _extract_validations_layers(self, dialog, participant_index: int):
        """
        Формирование слоев проверок
        """
        features = {}
        participant1_messages, participant2_messages = self._extract_dialog_messages(dialog)
        
        self.logger.info(f"participant1_messages {participant1_messages}")
        self.logger.info(f"participant2_messages {participant2_messages}")
        
        """
        Слой проверки с отправкой полного контекста диалога с последним сообщением трем моделям LLM
        """
        openai_response, gigachat_response = await asyncio.gather(
            self._validate_with_openai(dialog, participant_index),
            self._validate_with_gigachat(dialog, participant_index),
        )
        if openai_response != None:
            features['openai'] = openai_response
        if gigachat_response != None:
            features['gigachat'] = gigachat_response
        
        return features

    async def predict(self, dialog, participant_index: int):
        features = await self._extract_validations_layers(dialog, participant_index)
        
        weighted_score = sum(features[key] * self.weights.get(key, 1) for key in features)
    
        total_weight = sum(self.weights.values())
        score = weighted_score / total_weight if total_weight > 0 else 0
        
        return score