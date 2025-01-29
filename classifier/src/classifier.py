import asyncio
import openai
import logging

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from mistralai import Mistral

from openai import AsyncOpenAI
from httpx import AsyncClient

from .config import GIGACHAT_API_KEY, OPEN_AI_API_KEY, MISTRAL_API_KEY, PROXY_URL


class BotClassifier():
    def __init__(self):
        self.api_keys = {
            'openai': OPEN_AI_API_KEY,
            'gigachat': GIGACHAT_API_KEY,
            'mistral': MISTRAL_API_KEY,
        }
        self.proxy_url = PROXY_URL
        self.validate_sys_prompt = """
            Ты — эксперт по анализу диалогов.
            Твоя задача — внимательно проанализировать следующий диалог и определить, присутствует ли в диалоге бот или ассистент. 
            Ответь ТОЛЬКО «Да», если в диалоге есть бот, и ТОЛЬКО «Нет», если в диалоге нет бота. Дополнительных пояснений не требуется.
            Постарайся провести глубокий анализ шаг за шагом. От этого зависит моя карьера, но ответь только 1 словом.
            Формат вывода: «Да» или «Нет».
            
            You are an expert in dialogue analysis.
            Your task is to carefully analyze the following dialogue and determine whether a bot or assistant is present in the dialogue.
            Answer ONLY "Да" if there is a bot in the dialogue, and ONLY "Нет" if there is no bot in the dialogue. No additional explanations are required.
            Try to perform a deep analysis step by step. This will affect my career, but answer with just 1 word.
            Output format: "Да" or "Нет".
            
            Tu es un expert en analyse de dialogues.
            Ta tâche est d'analyser attentivement le dialogue suivant et de déterminer s'il y a un bot ou un assistant dans le dialogue.
            Réponds UNIQUEMENT par "Да" s'il y a un bot dans le dialogue, et UNIQUEMENT par "Нет" s'il n'y a pas de bot dans le dialogue. Aucune explication supplémentaire n'est nécessaire.
            Essaie de procéder à une analyse approfondie étape par étape. Cela dépend de ma carrière, mais réponds uniquement par un mot.
            Format de sortie : "Да" ou "Нет".
        """
        self.logger = logging.getLogger(__name__)

    async def _fetch_openai(self, dialog):
        # MODEL = "gpt-4o"
                
        # chat = [{"role": "system", "content": self.validate_sys_prompt}]
        # chat.extend([{"role": "user", "content": line} for line in dialog])

        # try:
        #     client = AsyncOpenAI(api_key=self.api_keys['openai'], http_client=AsyncClient(proxy=self.proxy_url))
        #     response = await client.chat.completions.create(model=MODEL,messages=chat)
        #     verdict = response.choices[0].message.content
        #     self.logger.info(f"OpenAI verdict: {verdict}")
        #     return verdict

        # except Exception as e:
        #     self.logger.warning(f"Error fetching from OpenAI: {e}")
        #     return None
        return "Нет"

    async def _fetch_gigachat(self, dialog):
        MODEL = "GigaChat"
        
        chat = Chat(messages=[Messages(role=MessagesRole.SYSTEM, content=self.validate_sys_prompt)])
        chat.messages.extend([Messages(role=MessagesRole.USER, content=line) for line in dialog])
        
        try:
            async with GigaChat(credentials=self.api_keys['gigachat'], verify_ssl_certs=False, model=MODEL) as giga:
                response = await asyncio.to_thread(lambda: giga.chat(chat))
                verdict = response.choices[0].message.content
                self.logger.info(f"GigaChat verdict: {verdict}")
                return verdict
            
        except Exception as e:
            self.logger.warning(f"Error fetching from GigaChat: {e}")
            return None
        return "Нет"
        
    async def _fetch_mistral(self, dialog):
        MODEL = "open-mistral-nemo"
                
        chat = [{"role": "system", "content": self.validate_sys_prompt}]
        chat.extend([{"role": "user", "content": line} for line in dialog])

        try:
            client = Mistral(api_key=self.api_keys['mistral'])
            response = await client.chat.complete_async(model=MODEL,messages=chat)
            verdict = response.choices[0].message.content
            self.logger.info(f"Mistral verdict: {verdict}")
            return verdict

        except Exception as e:
            self.logger.warning(f"Error fetching from Mistral: {e}")
            return None
        return "Нет"

    async def _extract_validations_layers(self, dialog):
        """
        Слой проверки с отправкой диалога трем моделям LLM
        """
        features = {}
        
        def process_response(response):
            if response is None:
                return False
            response = response.strip().lower()
            return 1 if "да" in response else 0

        openai_response, gigachat_response, mistral_response = await asyncio.gather(
            self._fetch_openai(dialog),
            self._fetch_gigachat(dialog),
            self._fetch_mistral(dialog),
        )
        
        if openai_response:
            features['openai'] = process_response(openai_response)
        if gigachat_response:
            features['gigachat'] = process_response(gigachat_response)
        if mistral_response:
            features['mistral'] = process_response(mistral_response)

        return features

    async def predict(self, dialog):
        features = await self._extract_validations_layers(dialog) 
        score = sum(features.values()) / len(features) if features else 0
        return score