"""
EVA 4.0 — Áudio (TTS + Push-to-Talk)
Edge-TTS para fala, keyboard para push-to-talk.
"""
import asyncio
import os
import re

import keyboard
import pygame

import edge_tts

from core.config import Config
from core.logger import Logger


class AudioService:
    """TTS via edge-tts com suporte a interrupção."""

    def __init__(self):
        # Aumentar tamanho do buffer para evitar jitter e inicializar na frequência padrão
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        self.voice = Config.VOICE
        self.is_playing = False
        self._temp_counter = 0

    async def speak(self, text: str, interrupt_event: asyncio.Event):
        """Fala o texto com motor assíncrono (Produtor/Consumidor) para eliminar delay antes do áudio e gaps entre frases."""
        if not text or not text.strip():
            return

        # Tocar 0.3s de silêncio (buffer limpo) para "acordar" o hardware de som
        try:
            import array
            silence_arr = array.array('h', [0] * 26460) # ~0.3s para 44100Hz estéreo
            pygame.mixer.Sound(buffer=silence_arr).play()
        except Exception:
            pass

        # Limpar texto para TTS
        clean = re.sub(r'[^\w\s,.!?:;\-\(\)áàâãéèêíìîóòôõúùûçÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ]', '', text)
        clean = clean.replace("*", "").replace("#", "").strip()
        if not clean:
            return

        # Corta a fala por frase para adiantar o download enquanto toca a atual
        sentences = re.split(r'(?<=[.!?]) +', clean)

        # Usamos uma Fila para o Player tocar os áudios que o Downloader já baixou
        q = asyncio.Queue()
        self._temp_counter += 1
        base_path = f"_tts_temp_{self._temp_counter}"

        async def downloader():
            for i, sentence in enumerate(sentences):
                if not sentence.strip() or interrupt_event.is_set():
                    break
                path = f"{base_path}_{i}.mp3"
                try:
                    communicate = edge_tts.Communicate(
                        sentence,
                        self.voice,
                        rate=Config.TTS_RATE,
                        pitch=Config.TTS_PITCH,
                    )
                    await communicate.save(path)
                    await q.put(path)
                except Exception as e:
                    Logger.err(f"Erro TTS DL: {e}")
            await q.put(None)  # Fim da linha

        async def player():
            while True:
                path = await q.get()
                if path is None or interrupt_event.is_set():
                    break
                await self._play(path, interrupt_event)
                try:
                    os.remove(path)
                except OSError:
                    pass

        # Dispara os dois simultaneamente. O Download libera a frase 1 em 0.5s, 
        # permitindo que o Player inicie enquanto a frase 2 baixa no background!
        await asyncio.gather(
            asyncio.create_task(downloader()),
            asyncio.create_task(player())
        )

    async def _play(self, path: str, interrupt_event: asyncio.Event):
        """Reproduz um arquivo de áudio decodificado evitando recortes nativos do Pygame."""
        try:
            if path.endswith("_0.mp3"):
                try:
                    import array
                    # Cria um ruído baixíssimo (frequência baixa) mais longo para abrir o gate
                    arr = array.array('h')
                    for i in range(48000): # ~1 segundo de buffer
                        val = 50 if (i // 100) % 2 == 0 else -50
                        arr.append(val)
                        arr.append(val)
                    silence = pygame.mixer.Sound(buffer=arr)
                    channel_silence = silence.play()
                    
                    # Espera 0.4s. O áudio do silence AINDA está tocando quando iniciarmos o principal.
                    # Isso garante que a placa Bluetooth não feche o noise gate.
                    await asyncio.sleep(0.4)
                except Exception as e:
                    Logger.err(f"Erro ao tocar wake-up: {e}")

            sound = pygame.mixer.Sound(path)
            channel = sound.play()
            self.is_playing = True

            while channel.get_busy():
                if interrupt_event.is_set():
                    channel.stop()
                    break
                await asyncio.sleep(0.05)
        except Exception:
            pass
        finally:
            self.is_playing = False

    def stop(self):
        """Para reprodução imediatamente."""
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        self.is_playing = False


class PushToTalk:
    """Gerencia o estado do push-to-talk via tecla configurável."""

    def __init__(self):
        self.key = Config.PTT_KEY
        self._was_pressed = False
        Logger.ptt(f"Push-to-Talk configurado na tecla: {self.key.upper()}")

    @property
    def is_pressed(self) -> bool:
        """Retorna True se a tecla PTT está pressionada agora."""
        try:
            return keyboard.is_pressed(self.key)
        except Exception:
            return False

    def just_pressed(self) -> bool:
        """Retorna True apenas no momento que a tecla foi pressionada (edge detection)."""
        pressed = self.is_pressed
        if pressed and not self._was_pressed:
            self._was_pressed = True
            return True
        if not pressed:
            self._was_pressed = False
        return False

    def just_released(self) -> bool:
        """Retorna True apenas no momento que a tecla foi solta."""
        pressed = self.is_pressed
        if not pressed and self._was_pressed:
            self._was_pressed = False
            return True
        if pressed:
            self._was_pressed = True
        return False
