import asyncio
import openai
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from mistralai import Mistral
from Levenshtein import ratio

from openai import AsyncOpenAI
from httpx import AsyncClient

from .config import GIGACHAT_API_KEY, OPEN_AI_API_KEY, PROXY_URL, VALIDATE_SYS_PROMPT


class BotClassifier():
    def __init__(self):
        self.api_keys = {
            'openai': OPEN_AI_API_KEY,
            'gigachat': GIGACHAT_API_KEY,
        }
        self.vectorizer = TfidfVectorizer()
        self.proxy_url = PROXY_URL
        self.validate_sys_prompt = VALIDATE_SYS_PROMPT
        self.logger = logging.getLogger(__name__)

    async def _fetch_openai(self, dialog):
        MODEL = "gpt-4o"
                
        chat = [{"role": "system", "content": self.validate_sys_prompt}]
        chat.extend([{"role": "user", "content": line} for line in dialog])

        try:
            client = AsyncOpenAI(api_key=self.api_keys['openai'], http_client=AsyncClient(proxy=self.proxy_url))
            response = await client.chat.completions.create(model=MODEL,messages=chat)
            verdict = response.choices[0].message.content.lower()
            self.logger.info(f"OpenAI verdict: {verdict}")
            return 1 if "да" in verdict else 0

        except Exception as e:
            self.logger.warning(f"Error fetching from OpenAI: {e}")
            return None
        return "Нет"

    async def _fetch_gigachat(self, dialog):
        MODEL = "GigaChat-Pro"
        
        chat = Chat(messages=[Messages(role=MessagesRole.SYSTEM, content=self.validate_sys_prompt)])
        chat.messages.extend([Messages(role=MessagesRole.USER, content=line) for line in dialog])
        
        try:
            async with GigaChat(credentials=self.api_keys['gigachat'], verify_ssl_certs=False, model=MODEL) as giga:
                response = await asyncio.to_thread(lambda: giga.chat(chat))
                verdict = response.choices[0].message.content.lower()
                self.logger.info(f"GigaChat verdict: {verdict}")
                return 1 if "да" in verdict else 0
            
        except Exception as e:
            self.logger.warning(f"Error fetching from GigaChat: {e}")
            return None
        return "Нет"
        
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
    
    def _check_mirroring(self, participant1_messages, participant2_messages):
        """
        Проверяет зеркальные ответы по косинусному сходству TF-IDF и расстоянию Левенштейна
        """
        if not participant1_messages or not participant2_messages:
            return 0
        
        mirror_count = 0
        total_pairs = min(len(participant1_messages), len(participant2_messages))
        
        for i in range(total_pairs):
            msg1 = participant1_messages[i]
            msg2 = participant2_messages[i]
            
            tfidf_matrix = self.vectorizer.fit_transform([msg1, msg2])
            cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            
            levenshtein_sim = ratio(msg1, msg2)
            
            self.logger.info(f"Pair {i}: Cosine={cosine_sim:.2f}, Levenshtein={levenshtein_sim:.2f}")
            
            if cosine_sim > 0.5 or levenshtein_sim > 0.5:
                mirror_count += 1
        
        return mirror_count / total_pairs if total_pairs > 0 else 0

    async def _extract_validations_layers(self, dialog):
        features = {}
        participant1_messages, participant2_messages = self._extract_dialog_messages(dialog)
        
        self.logger.info(f"participant1_messages {participant1_messages}")
        self.logger.info(f"participant2_messages {participant2_messages}")
        
        """
        Слой проверки с отправкой диалога трем моделям LLM
        """
        # openai_response, gigachat_response = await asyncio.gather(
        #     self._fetch_openai(dialog),
        #     self._fetch_gigachat(dialog),
        # )
        # if openai_response:
        #     features['openai'] = openai_response
        # if gigachat_response:
        #     features['gigachat'] = gigachat_response

        """
        Слой проверки на зеркалирования
        """
        mirroring_score = self._check_mirroring(participant1_messages, participant2_messages)
        features['mirroring'] = mirroring_score
        
        self.logger.info(f"Mirroring score: {mirroring_score}")
        
        return features

    async def predict(self, dialog):
        features = await self._extract_validations_layers(dialog)
        score = sum(features.values()) / len(features) if features else 0
        return score