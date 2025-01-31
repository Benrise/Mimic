import asyncio
import logging
import joblib

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
        self.validate_sys_prompt = VALIDATE_SYS_PROMPT
        self.model = joblib.load('./models/bot_classifier_model.pkl')
        self.vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')
        self.weights = {
            'model': 0.5, 
            'openai': 0.3,
            'gigachat': 0.2,
        }

    async def _validate_with_openai(self, dialog, participant_index):
        """
        Анализируем диалог для конкретного участника с помощью ChatGPT
        """
        model = OPENAI_VALIDATION_MODEL
        sys_prompt_validate_filled = VALIDATE_SYS_PROMPT.format(
            participant_index=participant_index
        )
        api_key = None
        base_url = None
        http_client = None

        if self.api_keys['bothub'] and BOTHUB_API_BASE_URL:
            self.logger.info("Using BotHub API")
            base_url = BOTHUB_API_BASE_URL
            api_key = self.api_keys['bothub']
        elif self.api_keys['openai'] and PROXY_URL:
            self.logger.info("Using Alex's OpenAI API")
            http_client = AsyncClient(proxy=PROXY_URL) if self.proxy_url else None
            api_key = self.api_keys['openai']
        else:
            self.logger.error("Neither BotHub API nor OpenAI API credentials found.")
            return 0.5

        chat = [{"role": "system", "content": sys_prompt_validate_filled}]
        chat.extend([{"role": "user", "content": line} for line in dialog])

        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
            )
            response = await client.chat.completions.create(model=model, messages=chat)
            verdict = response.choices[0].message.content.lower()
            self.logger.info(f"OpenAI verdict: {verdict}")
            return float(verdict)

        except Exception as e:
            self.logger.warning(f"Error fetching from OpenAI: {e}")
            return 0.5

    async def _validate_with_gigachat(self, dialog, participant_index):
        """
        Анализируем диалог для конкретного участника с помощью GigaChat 
        """
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

    async def _extract_validations_layers(self, message, dialog, participant_index: int):
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
        features['openai'] = openai_response
        features['gigachat'] = gigachat_response
            
        """
        Слой проверки обученной модели TF-IDF + Лог.регрессии
        """
        
        tfidf_features = self._get_tfidf_features(message)
        message_score = self._get_model_prediction(tfidf_features)
        features['model'] = message_score
        
        return features
    
    def _get_tfidf_features(self, message):
        """
        Преобразуем сообщение в TF-IDF векторы
        """
        return self.vectorizer.transform([message])

    def _get_model_prediction(self, tfidf_features):
        """
        Предсказание модели на основе TF-IDF признаков
        """
        verdict = self.model.predict(tfidf_features)
        self.logger.info(f'Model verdict: {verdict}')
        return verdict[0]

    def _calculate_final_score(self, features):
        """
        Вычисление итогового балла на основе взвешенных оценок
        """
        weighted_score = sum(features[key] * self.weights.get(key, 1) for key in features)
        total_weight = sum(self.weights.values())
        return weighted_score / total_weight if total_weight > 0 else 0

    async def predict(self, message, dialog, participant_index: int):
        """
        Основная функция для предсказания
        """
        features = await self._extract_validations_layers(message, dialog, participant_index)
        
        score = self._calculate_final_score(features)
        
        return score