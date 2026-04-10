"""
EVA 4.0 — Memória Semântica (ONNX Runtime GPU)
Embeddings locais sem PyTorch usando ONNX Runtime.
"""
import json
import os
from datetime import datetime

import numpy as np

from core.config import Config
from core.logger import Logger


class MemoryService:
    """Memória de longo prazo com busca semântica via ONNX."""

    def __init__(self):
        self.storage_path = Config.MEMORY_FILE
        self.memories = self._load_memories()

        Logger.mem("Carregando modelo de embeddings (ONNX)...")
        self._init_embeddings()
        Logger.mem(f"Embeddings prontos. {len(self.memories)} memórias carregadas.")

    # ── Embeddings ONNX ──────────────────────────────────────
    def _init_embeddings(self):
        """Inicializa ONNX Runtime com all-MiniLM-L6-v2."""
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            # Download / cache do modelo
            model_dir = self._ensure_model()
            model_path = os.path.join(model_dir, "model.onnx")
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")

            # Tentar GPU primeiro, fallback CPU
            providers = []
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            self.session = ort.InferenceSession(model_path, providers=providers)
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
            self.tokenizer.enable_truncation(max_length=128)

            active = self.session.get_providers()
            Logger.mem(f"ONNX providers: {active}")

        except Exception as e:
            Logger.err(f"Erro ao inicializar embeddings: {e}")
            self.session = None
            self.tokenizer = None

    def _ensure_model(self) -> str:
        """Baixa o modelo ONNX se necessário. Retorna o caminho local."""
        cache_dir = os.path.join(Config.EMBEDDINGS_CACHE, "all-MiniLM-L6-v2-onnx")

        model_path = os.path.join(cache_dir, "model.onnx")
        tokenizer_path = os.path.join(cache_dir, "tokenizer.json")

        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            return cache_dir

        Logger.mem("Baixando modelo ONNX (primeira execução)...")
        os.makedirs(cache_dir, exist_ok=True)

        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="onnx/model.onnx",
            local_dir=cache_dir,
        )
        # Move from onnx/ subfolder
        onnx_sub = os.path.join(cache_dir, "onnx", "model.onnx")
        if os.path.exists(onnx_sub):
            import shutil
            shutil.move(onnx_sub, model_path)
            shutil.rmtree(os.path.join(cache_dir, "onnx"), ignore_errors=True)

        hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="tokenizer.json",
            local_dir=cache_dir,
        )
        # Move if in subfolder
        tok_sub = os.path.join(cache_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path) and os.path.exists(tok_sub):
            import shutil
            shutil.move(tok_sub, tokenizer_path)

        Logger.mem("Modelo ONNX baixado com sucesso!")
        return cache_dir

    def _encode(self, text: str) -> np.ndarray:
        """Gera embedding para um texto."""
        if self.session is None:
            return np.zeros(384)

        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # Mean pooling
        token_embeddings = outputs[0]  # (1, seq_len, hidden_size)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counted = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embedding = (summed / counted).flatten()

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    # ── Storage ──────────────────────────────────────────────
    def _load_memories(self) -> list:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception):
                return []
        return []

    def _save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    # ── Public API ───────────────────────────────────────────
    async def add_fact(self, fact: str) -> str:
        """Memoriza um fato com embedding."""
        try:
            embedding = self._encode(fact).tolist()
            self.memories.append({
                "fact": fact,
                "embedding": embedding,
                "ts": datetime.now().isoformat(),
            })
            self._save()
            Logger.mem(f"Memorizado: '{fact}'")
            return f"Memorizado: '{fact}'"
        except Exception as e:
            Logger.err(f"Erro ao memorizar: {e}")
            return "Erro ao memorizar."

    async def get_relevant(self, query: str, top_k: int = 3, threshold: float = 0.6) -> str:
        """Busca memórias relevantes por similaridade."""
        if not self.memories or self.session is None:
            return ""

        try:
            qv = self._encode(query)
            scored = []
            for m in self.memories:
                mv = np.array(m["embedding"])
                score = float(np.dot(qv, mv))  # Já normalizado
                scored.append((score, m["fact"]))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = [s[1] for s in scored[:top_k] if s[0] > threshold]

            if results:
                Logger.mem(f"Recuperado: {results}")
            return "\n".join(results)

        except Exception as e:
            Logger.err(f"Erro na busca de memória: {e}")
            return ""
