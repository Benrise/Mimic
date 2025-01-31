import joblib
import logging


class ModelClassifier():
    def __init__(self, model_path='./models/bot_classifier_model.pkl', vectorizer_path='./models/tfidf_vectorizer.pkl'):
        self.logger = logging.getLogger(__name__)
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.logger.info("Model and vectorizer loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model or vectorizer: {e}")
            raise
        
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
    
    def predict(self, message):
        """
        Определяет, является ли сообщение ботоподобным (1) или человеческим (0).
        """
        tfidf_features = self._get_tfidf_features(message)
        verdict = self.model.predict(tfidf_features)[0]
        self.logger.info(f"Model verdict: {verdict}")
        return verdict