import os
import re
import json
from django.conf import settings

class MedicalRAGEngine:
    def __init__(self):
        self.corpus_path = os.path.join(settings.BASE_DIR, 'core_ml', 'rag_corpus.json')
        self.corpus = []
        if os.path.exists(self.corpus_path):
            try:
                with open(self.corpus_path, 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f)
            except Exception as e:
                print(f"[RAG Engine Error] Failed to load corpus: {e}")
        else:
            print(f"[RAG Engine Warning] Corpus file not found at: {self.corpus_path}")

    def retrieve(self, query: str, modality: str, limit: int = 2) -> list:
        """
        Query the local medical knowledge database, filter by modality,
        and return the top matching text chunks based on keyword matching score.
        """
        if not self.corpus:
            return []

        # Filter by modality (CT, CXR, ECG, BLOOD_TEST)
        filtered_corpus = [chunk for chunk in self.corpus if chunk['modality'] == modality]
        if not filtered_corpus:
            filtered_corpus = self.corpus # fallback
            
        # Tokenize query keywords
        query_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower())) # terms longer than 2 chars
        
        scored_chunks = []
        for chunk in filtered_corpus:
            content_text = f"{chunk['condition']} {chunk['title']} {chunk['content']}".lower()
            content_words = re.findall(r'[a-zA-Z]{3,}', content_text)
            
            # Score calculated as matching keyword frequencies
            score = 0
            for word in query_words:
                tf = content_words.count(word)
                if tf > 0:
                    # Give extra weight if keyword appears in condition tag or title
                    boost = 1.0
                    if word in chunk['condition'].lower():
                        boost = 5.0
                    elif word in chunk['title'].lower():
                        boost = 3.0
                    score += tf * boost
                    
            scored_chunks.append((score, chunk))
            
        # Sort by match score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Extract top matching chunks
        results = []
        for score, chunk in scored_chunks[:limit]:
            # Only return chunks that have some correlation or default to top entries
            results.append(chunk)
            
        return results

_engine = None

def get_rag_engine() -> MedicalRAGEngine:
    global _engine
    if _engine is None:
        _engine = MedicalRAGEngine()
    return _engine
