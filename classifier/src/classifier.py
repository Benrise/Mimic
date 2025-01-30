import asyncio
import logging
import statistics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio
from language_tool_python import LanguageTool

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
        self.grammar_tool = LanguageTool('ru-RU')
        self.vectorizer = TfidfVectorizer()
        self.proxy_url = PROXY_URL
        self.validate_sys_prompt = VALIDATE_SYS_PROMPT
        self.weights = {
            'openai': 0.4,
            'gigachat': 0.4,
            'mirroring': 0.2,
            'grammar': 0.3,
            'length': 0.1
        }

    async def _validate_with_openai(self, dialog):
        model = OPENAI_VALIDATION_MODEL
        base_url = None
        
        if BOTHUB_API_BASE_URL or len(BOTHUB_API_BASE_URL) > 0:
            base_url = BOTHUB_API_BASE_URL
 
        chat = [{"role": "system", "content": self.validate_sys_prompt}]
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
            return 1 if "да" in verdict else 0

        except Exception as e:
            self.logger.warning(f"Error fetching from OpenAI: {e}")
            return None

    async def _validate_with_gigachat(self, dialog):
        model = GIGACHAT_VALIDATION_MODEL
        
        chat = Chat(messages=[Messages(role=MessagesRole.SYSTEM, content=self.validate_sys_prompt)])
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
                return 1 if "да" in verdict else 0
            
        except Exception as e:
            self.logger.warning(f"Error fetching from GigaChat: {e}")
            return None
        
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
        if not participant1_messages or not participant2_messages or len(participant1_messages) < 2 or len(participant2_messages) < 2:
            return 0.5
        
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
        
        return mirror_count / total_pairs if total_pairs > 0 else 0.5
    
    
    def _check_grammar_errors(self, participant1_messages, participant2_messages):
        """
        Проверяет сообщения на грамматические ошибки
        """
        def count_errors(messages):
            return sum(len(self.grammar_tool.check(msg)) for msg in messages)
        
        errors_p1 = count_errors(participant1_messages)
        errors_p2 = count_errors(participant2_messages)
        
        self.logger.info(f"Grammar errors - P1: {errors_p1}, P2: {errors_p2}")
        
        if (errors_p1 == 0 and errors_p2 > 0) or (errors_p1 > 0 and errors_p2 == 0):
            return 1.0
        return 0.5
    

    def _check_message_length(self, participant1_messages, participant2_messages):
        """
        Проверяет и сравнивает длины слов и предложений
        """
        def analyze_text(messages):
            sentence_lengths = []
            word_lengths = []
            
            for message in messages:
                sentences = message.split('.')
                for sentence in sentences:
                    words = sentence.split()
                    sentence_lengths.append(len(words))
                    word_lengths.extend([len(word) for word in words])
            
            avg_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 0
            avg_word_length = statistics.mean(word_lengths) if word_lengths else 0
            return avg_sentence_length, avg_word_length
        
        avg_sent_len_p1, avg_word_len_p1 = analyze_text(participant1_messages)
        avg_sent_len_p2, avg_word_len_p2 = analyze_text(participant2_messages)
        
        self.logger.info(f"Avg sentence words - P1: {avg_sent_len_p1:.2f}, P2: {avg_sent_len_p2:.2f}")
        self.logger.info(f"Avg word length - P1: {avg_word_len_p1:.2f}, P2: {avg_word_len_p2:.2f}")
        
        return 1.0 if abs(avg_sent_len_p1 - avg_sent_len_p2) > 5 or abs(avg_word_len_p1 - avg_word_len_p2) > 2 else 0.5

    async def _extract_validations_layers(self, dialog):
        """
        Формирование слоев проверок
        """
        features = {}
        participant1_messages, participant2_messages = self._extract_dialog_messages(dialog)
        
        self.logger.info(f"participant1_messages {participant1_messages}")
        self.logger.info(f"participant2_messages {participant2_messages}")
        
        """
        Слой проверки с отправкой диалога трем моделям LLM
        """
        openai_response, gigachat_response = await asyncio.gather(
            self._validate_with_openai(dialog),
            self._validate_with_gigachat(dialog),
        )
        if openai_response != None:
            features['openai'] = openai_response
        if gigachat_response != None:
            features['gigachat'] = gigachat_response

        """
        Слой проверки на зеркалирования
        """
        mirroring_score = self._check_mirroring(participant1_messages, participant2_messages)
        features['mirroring'] = mirroring_score
        
        self.logger.info(f"Mirroring score: {mirroring_score}")
    
        """
        Слой проверки на наличие грамматических ошибок
        """
        grammar_score = self._check_grammar_errors(participant1_messages, participant2_messages)
        features['grammar'] = grammar_score
        
        self.logger.info(f"Grammar score: {grammar_score}")
        
        """
        Слой проверки на длину слов и предложений
        """
        length_score = self._check_message_length(participant1_messages, participant2_messages)
        features['length'] = length_score
        
        self.logger.info(f"Length score: {length_score}")
        
        self.logger.info(f"Assembleded features for all validation layers: {features}")
        
        return features

    async def predict(self, dialog):
        features = await self._extract_validations_layers(dialog)
        
        weighted_score = sum(features[key] * self.weights.get(key, 1) for key in features)
    
        total_weight = sum(self.weights.values())
        score = weighted_score / total_weight if total_weight > 0 else 0
        
        return score