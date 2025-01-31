import logging
from .classifier import BotClassifier

logger = logging.getLogger(__name__)

_classifier = None


def load_classifier():
    global _classifier
    if _classifier is None:
        logger.info("Loading classifier...")
        _classifier = BotClassifier()
    return _classifier


def format_conversation(messages):
    """
    Формирует строки диалога для последующего анализа.
    Пример:
        "0: Привет\n1: Здравствуйте!\n..."
    """
    return "\n".join(
        [f"{msg['participant_index']}: {msg['text']}" for msg in messages]
    )

def format_conversation_local(messages):
    result = []
    
    for i in range(1, len(messages)):
        if (messages[i].role == 'assistant'):
             result.append(f"1: {messages[i].content}")
        else:
            result.append(f"0: {messages[i].content}")
            
    return "\n".join(result)
             

def classify_text(messages, participant_index: int) -> float:
    """
    Прогоняет диалог через zero-shot классификатор и возвращает
    вероятность, что в диалоге есть бот.
    """
    classifier = load_classifier()
    conversation_text = format_conversation(messages)
    
    logger.info(f"Conversation to classify: {conversation_text}")
    
    result = classifier.predict(conversation_text, participant_index)
    
    logger.info(f"Conversation classified: {result}")

    return result


def classify_dialog(messages: list, participant_index: int) -> float:
    classifier = load_classifier()
    conversation_text = format_conversation_local(messages)
    
    logger.info(f"Conversation to classify: {conversation_text}")
    
    result = classifier.predict(conversation_text, participant_index)

    return result