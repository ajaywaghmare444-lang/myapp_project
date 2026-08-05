import re
import logging
from typing import List
from sentence_transformers import CrossEncoder

logger = logging.getLogger("app.services.reranker_service")

class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading CrossEncoder model '{self.model_name}'...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank_chunks(self, query: str, full_text: str, top_k: int = 2) -> str:
        """
        Splits full_text into individual chunks, reranks them against the query
        using the CrossEncoder model, and returns the top_k best chunks.
        """
        if not full_text or not full_text.strip():
            return full_text

        # Parse individual chunks from full_text by [Source: ...], bullet headers, or double newlines
        raw_chunks = [c.strip() for c in re.split(r'\n(?=\[Source:|\n)', full_text) if c.strip()]
        
        if len(raw_chunks) <= top_k:
            logger.info(f"Retrieved {len(raw_chunks)} chunks, which is <= top_k ({top_k}). Returning all chunks.")
            return full_text

        logger.info(f"Reranking {len(raw_chunks)} retrieved chunks using CrossEncoder model '{self.model_name}'...")
        
        try:
            model = self._get_model()
            pairs = [[query, chunk] for chunk in raw_chunks]
            scores = model.predict(pairs)

            # Zip scores with chunks and sort descending
            scored_chunks = sorted(zip(scores, raw_chunks), key=lambda x: float(x[0]), reverse=True)
            
            logger.info("=== CROSS-ENCODER RERANKING SCORES ===")
            for rank, (score, chunk) in enumerate(scored_chunks, 1):
                preview = chunk.replace('\n', ' ')[:100]
                logger.info(f"  Rank {rank} (Score: {float(score):.4f}): {preview}...")
            logger.info("=======================================")

            # Select top_k best chunks
            best_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
            logger.info(f"Selected top {len(best_chunks)} best chunks out of {len(raw_chunks)}.")

            return "\n\n".join(best_chunks)
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to original chunks.", exc_info=True)
            return full_text

reranker_service = RerankerService()
