import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


class FaissSearchEngine:
    def __init__(self, model_path):
        # Используем стандартную предобученную модель
        self.model = SentenceTransformer(model_path)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.data = None


    def create_index(self, df, text_column='full_name', index_path='faiss_index.bin'):
        """Создание FAISS индекса из DataFrame"""
        embeddings = self.model.encode(
            df[text_column].tolist(), 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        faiss.write_index(self.index, index_path)
        df.to_pickle('data.pkl')
        print(f"Index created and saved to {index_path}")

    def load_index(self, index_path='faiss_index.bin', data_file='data.pkl'):
        """Загрузка предварительно сохраненного индекса"""
        self.index = faiss.read_index(index_path)
        self.data = pd.read_pickle(data_file)
        print("Index and data loaded successfully")

    def vectorize_query(self, query):
        """Векторизация текстового запроса"""
        return self.model.encode([query], convert_to_numpy=True).astype('float32')

    def search(self, query, top_n=5):
        """Поиск топ-N результатов по запросу"""
        if self.index is None or self.data is None:
            raise ValueError("Index and data not loaded. Call load_index() first")
            
        # Векторизация запроса
        query_vector = self.vectorize_query(query)
        
        # Поиск в индексе
        distances, indices = self.index.search(query_vector, top_n)
        
        # Формирование результатов
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS возвращает -1 для недостающих элементов
                results.append({
                    'full_name': self.data.iloc[idx]['full_name'],
                    'id': int(self.data.iloc[idx]['id']),
                    'Цена': self.data.iloc[idx]['Цена'],
                    'Заказов_за_месяц': int(self.data.iloc[idx]['Заказов_за_месяц']),
                    'Персональный_кэшбэк': self.data.iloc[idx]['Персональный_кэшбэк'],
                    'Время доставки': self.data.iloc[idx]['Время_доставки'],
                    'Рейтинг': str(self.data.iloc[idx]['Рейтинг'])

                })
        return results
