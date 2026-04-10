"""
EVA 4.0 — Toolbox Extensível
Ferramentas de ação que a EVA pode executar via tool-calling.
"""
import asyncio
import os
import webbrowser

import pyautogui
import pywhatkit

from core.logger import Logger


# ── Tool Registry ────────────────────────────────────────────
# Adicione novas ferramentas aqui: basta criar a função async e 
# adicionar ao TOOLS dict + TOOLS_DEFINITIONS list.

APP_ALIASES = {
    "bloco de notas": "notepad",
    "notepad": "notepad",
    "navegador": "chrome",
    "chrome": "chrome",
    "calculadora": "calc",
    "calc": "calc",
    "vlc": "vlc",
    "explorer": "explorer",
    "terminal": "wt",
    "vscode": "code",
    "spotify": "spotify",
}


async def tool_click(text: str) -> str:
    """Clica em texto visível na tela via OCR."""
    # Importar sob demanda para evitar dependência circular
    from modules.vision import VisionService
    vision = VisionService()
    return await vision.find_and_click(text)


async def tool_open_url(url: str) -> str:
    """Abre uma URL no navegador padrão."""
    if not url.startswith("http"):
        url = "https://" + url
    await asyncio.to_thread(webbrowser.open, url)
    Logger.tool(f"Abri: {url}")
    return f"Abri {url}."


async def tool_open_app(app_name: str) -> str:
    """Abre um aplicativo do Windows."""
    cmd = APP_ALIASES.get(app_name.lower(), app_name.lower())
    await asyncio.to_thread(os.system, f"start {cmd}")
    Logger.tool(f"Abri: {app_name}")
    return f"Abri o {app_name}."


async def tool_close_app(app_name: str) -> str:
    """Fecha um aplicativo do Windows."""
    exe_map = {
        "bloco de notas": "notepad.exe",
        "navegador": "chrome.exe",
        "chrome": "chrome.exe",
        "calculadora": "Calculator.exe",
        "vlc": "vlc.exe",
        "spotify": "Spotify.exe",
    }
    exe = exe_map.get(app_name.lower(), app_name.lower())
    if not exe.endswith(".exe"):
        exe += ".exe"
    await asyncio.to_thread(os.system, f"taskkill /F /IM {exe}")
    Logger.tool(f"Fechei: {app_name}")
    return f"Fechei o {app_name}."


async def tool_type_text(text: str) -> str:
    """Digita texto no campo focado atualmente."""
    await asyncio.sleep(0.3)
    await asyncio.to_thread(pyautogui.write, text, interval=0.01)
    Logger.tool("Texto digitado.")
    return "Texto digitado."


async def tool_press_key(key: str) -> str:
    """Pressiona uma tecla ou combinação (ex: 'enter', 'ctrl+s')."""
    await asyncio.to_thread(pyautogui.hotkey, *key.split("+"))
    Logger.tool(f"Tecla: {key}")
    return f"Pressionei {key}."


async def tool_play_youtube(query: str) -> str:
    """Pesquisa e reproduz no YouTube."""
    await asyncio.to_thread(pywhatkit.playonyt, query)
    Logger.tool(f"YouTube: {query}")
    return f"Tocando {query} no YouTube!"


# ── Definitions (OpenAI Tool Format) ────────────────────────
TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Clica em um texto visível na tela do computador",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Texto para clicar"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Abre uma URL no navegador padrão",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL para abrir"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Abre um aplicativo do Windows (ex: bloco de notas, chrome, calculadora, vscode, terminal)",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string", "description": "Nome do aplicativo"}},
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_app",
            "description": "Fecha um aplicativo do Windows",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string", "description": "Nome do aplicativo"}},
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Digita texto no campo/janela atualmente focado",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Texto para digitar"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Pressiona tecla ou combinação de teclas (ex: enter, ctrl+s, alt+f4)",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "Tecla ou combinação"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_yt",
            "description": "Pesquisa e toca um vídeo no YouTube",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "O que procurar"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_mem",
            "description": "Salva uma informação importante na memória de longo prazo para lembrar depois",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "Fato para memorizar"}},
                "required": ["fact"],
            },
        },
    },
]


class ToolBox:
    """Registry de ferramentas executáveis pela EVA."""

    def __init__(self, memory_service=None):
        self.tools = {
            "click": tool_click,
            "open_url": tool_open_url,
            "open_app": tool_open_app,
            "close_app": tool_close_app,
            "type_text": tool_type_text,
            "press_key": tool_press_key,
            "play_yt": tool_play_youtube,
        }

        # Registrar save_mem se memory_service disponível
        if memory_service:
            self.tools["save_mem"] = memory_service.add_fact

    def get_definitions(self) -> list:
        """Retorna definições no formato OpenAI tool-calling."""
        return TOOLS_DEFINITIONS

    async def execute(self, name: str, args: dict) -> str:
        """Executa uma ferramenta pelo nome."""
        if name not in self.tools:
            Logger.err(f"Ferramenta desconhecida: {name}")
            return f"Ferramenta '{name}' não encontrada."

        Logger.tool(f"Executando: {name}({args})")
        try:
            result = await self.tools[name](**args)
            return str(result)
        except Exception as e:
            Logger.err(f"Erro em {name}: {e}")
            return f"Erro ao executar {name}: {e}"
