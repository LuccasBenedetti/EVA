"""
EVA 4.0 — Transcrição (faster-whisper + Push-to-Talk)
Grava áudio apenas quando PTT está ativo, transcreve no release.
"""
import asyncio
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy import signal

from core.config import Config
from core.logger import Logger


class TranscriptionService:
    """Transcrição via faster-whisper integrada com push-to-talk."""

    def __init__(self):
        Logger.sys(f"Carregando Whisper ({Config.WHISPER_MODEL})...")

        from faster_whisper import WhisperModel

        # Auto-detect CUDA (CTranslate2 tem backend próprio, não precisa de PyTorch)
        try:
            self.model = WhisperModel(
                Config.WHISPER_MODEL, device="cuda", compute_type="float16"
            )
            Logger.sys("Whisper carregado com CUDA.")
        except Exception:
            Logger.sys("CUDA indisponível para Whisper, usando CPU...")
            self.model = WhisperModel(
                Config.WHISPER_MODEL, device="cpu", compute_type="int8"
            )

        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.is_listening = True
        self.input_rate = Config.TARGET_RATE

    async def record_loop(self):
        """Loop de gravação contínuo do microfone."""
        device_info = sd.query_devices(kind="input")
        self.input_rate = int(device_info["default_samplerate"])
        Logger.sys(f"Microfone: {device_info['name']} @ {self.input_rate}Hz")

        def callback(indata, frames, time_info, status):
            self.audio_queue.put_nowait(indata.copy())

        with sd.InputStream(
            samplerate=self.input_rate, channels=Config.CHANNELS, callback=callback
        ):
            while self.is_listening:
                await asyncio.sleep(0.1)

    async def ptt_transcribe_loop(self, ptt, audio_service, interrupt_event, on_text_callback):
        """
        Loop de transcrição integrado com push-to-talk.
        Grava enquanto PTT pressionado, transcreve no release.
        """
        buffer = []
        pre_buffer = deque(maxlen=60)  # Guarda ~2.0s de áudio anterior
        is_recording = False
        status_printed = False

        while self.is_listening:
            # Drenar fila de áudio
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                chunk = None

            ptt_active = ptt.is_pressed

            if ptt_active:
                # Barge-in: se EVA está falando e PTT pressionado, interrompe
                if audio_service.is_playing:
                    Logger.sys("Interrupção de fala (barge-in)")
                    interrupt_event.set()
                    await asyncio.sleep(0.05)
                    interrupt_event.clear()

                if not is_recording:
                    is_recording = True
                    buffer = list(pre_buffer)  # Inclui pre-buffer
                    Logger.ptt("🎙️  Ouvindo...")
                    status_printed = True

                if chunk is not None:
                    buffer.append(chunk)

            else:
                if is_recording and buffer:
                    # PTT solto — transcrever
                    is_recording = False
                    Logger.ptt("🧠 Processando fala...")

                    audio_data = np.concatenate(buffer).flatten()

                    # Resample se necessário
                    if self.input_rate != Config.TARGET_RATE:
                        num_samples = int(
                            len(audio_data) * Config.TARGET_RATE / self.input_rate
                        )
                        audio_data = signal.resample(audio_data, num_samples)

                    # Transcrever
                    try:
                        segments, _ = self.model.transcribe(
                            audio_data, beam_size=5, language="pt", vad_filter=True
                        )
                        text = "".join([s.text for s in segments]).strip()

                        if text:
                            Logger.ptt(f"✅ \"{text}\"")
                            await on_text_callback(text)
                        else:
                            Logger.ptt("(nenhuma fala detectada)")
                    except Exception as e:
                        Logger.err(f"Erro na transcrição: {e}")

                    buffer = []
                    status_printed = False

                elif is_recording:
                    is_recording = False
                    status_printed = False

                # Manter pre-buffer
                if chunk is not None:
                    pre_buffer.append(chunk)

                # Status idle
                if not status_printed:
                    ptt_key = Config.PTT_KEY.upper()
                    print(f"\r  🔇 Pressione {ptt_key} para falar", end="", flush=True)
                    status_printed = True

    def stop(self):
        self.is_listening = False
