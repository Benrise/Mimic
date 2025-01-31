import time
import random
import logging

logger = logging.getLogger(__name__)


def human_sleep(message: str):
    delay_time = len(message) * 0.01
    logger.info(f"Applying delay: {delay_time} seconds")
    
    time.sleep(delay_time) 


def introduce_typos(text, typos=1):
    length = len(text)
    num_typos = length * typos // 40
    
    words = text.split()
    
    for _ in range(num_typos):
        word_index = random.randint(0, len(words) - 1)
        words[word_index] = random_typo(words[word_index])
    
    return ' '.join(words)


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
