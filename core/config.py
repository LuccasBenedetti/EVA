"""
EVA 4.0 — Configuração Centralizada
Carrega do .env e fornece defaults sensíveis.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuração global da EVA. Todos os módulos importam daqui."""

    # ── vLLM (WSL2) ─────────────────────────────────────────
    VLLM_WSL_DISTRO = os.getenv("VLLM_WSL_DISTRO", "Ubuntu")
    VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    VLLM_DTYPE = os.getenv("VLLM_DTYPE", "half")
    VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
    VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    VLLM_GPU_UTIL = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
    VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"

    # ── Whisper ──────────────────────────────────────────────
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

    # ── Áudio ────────────────────────────────────────────────
    TARGET_RATE = 16000
    CHANNELS = 1
    SILENCE_TIMEOUT = float(os.getenv("SILENCE_TIMEOUT", "0.6"))
    VOICE = os.getenv("VOICE", "pt-BR-ThalitaNeural")
    TTS_RATE = os.getenv("TTS_RATE", "+10%")
    TTS_PITCH = os.getenv("TTS_PITCH", "+5Hz")

    # ── Push-to-Talk ─────────────────────────────────────────
    PTT_KEY = os.getenv("PUSH_TO_TALK_KEY", "f2")

    # ── Memória ──────────────────────────────────────────────
    MEMORY_FILE = os.getenv("MEMORY_FILE", "eva_memory.json")
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDINGS_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    # ── Visão ────────────────────────────────────────────────
    VISION_INTERVAL = float(os.getenv("VISION_INTERVAL", "5.0"))
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "pt-BR")

    # ── LLM ──────────────────────────────────────────────────
    MAX_HISTORY = 15
    MAX_TOOL_ROUNDS = 5
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

    # ── CUDA Paths (Windows venv) ────────────────────────────
    @staticmethod
    def setup_cuda_paths():
        """Adiciona DLLs CUDA do venv ao PATH do Windows."""
        venv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "Lib", "site-packages", "nvidia")
        for lib in ["cublas", "cudnn", "cuda_nvrtc"]:
            bin_path = os.path.join(venv_path, lib, "bin")
            if os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
