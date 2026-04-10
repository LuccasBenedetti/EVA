"""
EVA 4.0 — Visão (OCR Windows Nativo)
Leitura de tela via WinSDK OCR, sem dependências externas.
"""
import asyncio
import os

import pyautogui

from core.config import Config
from core.logger import Logger

# Windows OCR SDK
from winsdk.windows.media.ocr import OcrEngine
from winsdk.windows.graphics.imaging import BitmapDecoder
from winsdk.windows.storage import StorageFile, FileAccessMode
from winsdk.windows.globalization import Language

# Engine singleton
_OCR_LANG = Language(Config.OCR_LANGUAGE)
_OCR_ENGINE = OcrEngine.try_create_from_language(_OCR_LANG)

_SCREENSHOT_PATH = os.path.abspath("_ocr_temp.png")


class VisionService:
    """OCR nativo do Windows para leitura de tela."""

    def __init__(self):
        self.current_text = ""
        self._running = True

    async def read_screen(self) -> str:
        """Captura a tela e extrai texto via OCR."""
        try:
            await asyncio.to_thread(pyautogui.screenshot().save, _SCREENSHOT_PATH)

            file = await StorageFile.get_file_from_path_async(_SCREENSHOT_PATH)
            stream = await file.open_async(FileAccessMode.READ)
            decoder = await BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            result = await _OCR_ENGINE.recognize_async(bitmap)

            lines = sorted(
                [
                    f"[Y:{int(l.words[0].bounding_rect.y):04}] {l.text}"
                    for l in result.lines
                    if l.words
                ]
            )
            text = "\n".join(lines)
            return text if text else "Tela vazia ou texto não detectado."

        except Exception as e:
            Logger.err(f"Erro OCR: {e}")
            return "Erro ao ler tela."

    async def find_and_click(self, target_text: str) -> str:
        """Encontra texto na tela e clica nele."""
        Logger.vis(f"Procurando '{target_text}' na tela...")
        try:
            await asyncio.to_thread(pyautogui.screenshot().save, _SCREENSHOT_PATH)

            file = await StorageFile.get_file_from_path_async(_SCREENSHOT_PATH)
            stream = await file.open_async(FileAccessMode.READ)
            decoder = await BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            result = await _OCR_ENGINE.recognize_async(bitmap)

            for line in result.lines:
                if target_text.lower() in line.text.lower():
                    words = list(line.words)
                    cx = (
                        words[0].bounding_rect.x
                        + (words[-1].bounding_rect.x + words[-1].bounding_rect.width)
                    ) / 2
                    cy = words[0].bounding_rect.y + words[0].bounding_rect.height / 2
                    await asyncio.to_thread(pyautogui.click, cx, cy)
                    return f"Cliquei em '{line.text}'."

            return f"Não encontrei '{target_text}' na tela."

        except Exception as e:
            Logger.err(f"Erro ao clicar: {e}")
            return "Erro ao procurar texto na tela."

    async def update_loop(self):
        """Atualiza a visão periodicamente em background."""
        while self._running:
            self.current_text = await self.read_screen()
            await asyncio.sleep(Config.VISION_INTERVAL)

    def stop(self):
        self._running = False
