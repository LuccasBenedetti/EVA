"""
EVA 4.0 — Assistente Virtual Modular
Orquestrador principal que integra todos os módulos.

Uso: python eva.py
Push-to-Talk: Segure F2 para falar, solte para a EVA processar.
"""
import asyncio
import json
import sys

from core.config import Config
from core.logger import Logger


async def main():
    # ── Banner ───────────────────────────────────────────────
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║         EVA 4.0 — Push-to-Talk           ║")
    print("  ║   Segure F2 para falar · Solte para EVA  ║")
    print("  ╚══════════════════════════════════════════╝")
    print()

    # ── Setup CUDA Paths ─────────────────────────────────────
    Config.setup_cuda_paths()

    # ── Boot vLLM ────────────────────────────────────────────
    from modules.llm import InferenceManager, LLMClient

    inf_manager = InferenceManager()
    if not await inf_manager.boot_sequence():
        Logger.err("vLLM não disponível. Abortando.")
        return

    # ── Inicializar Módulos ──────────────────────────────────
    Logger.sys("Inicializando módulos...")

    from modules.memory import MemoryService
    from modules.audio import AudioService, PushToTalk
    from modules.transcription import TranscriptionService
    from modules.vision import VisionService
    from modules.tools import ToolBox

    memory = MemoryService()
    audio = AudioService()
    ptt = PushToTalk()
    transcription = TranscriptionService()
    vision = VisionService()
    toolbox = ToolBox(memory_service=memory)

    llm = LLMClient(tools_definitions=toolbox.get_definitions())
    interrupt_event = asyncio.Event()

    Logger.sys("Todos os módulos prontos!")
    print()

    # ── Handler de Comando ───────────────────────────────────
    async def handle_command(text: str):
        """Processa um comando de voz transcrito."""
        # Interromper fala atual se estiver falando
        if audio.is_playing:
            interrupt_event.set()
            await asyncio.sleep(0.05)
            interrupt_event.clear()

        # Buscar memória relevante
        relevant_mem = await memory.get_relevant(text)

        # Montar contexto extra
        context_parts = []
        if relevant_mem:
            context_parts.append(f"[MEMÓRIA RELEVANTE]: {relevant_mem}")
        if vision.current_text:
            clean_vision = vision.current_text
            for bad_str in ["Pressione F2", "[MIC]", "[SISTEMA]", "Ouvindo..."]:
                clean_vision = clean_vision.replace(bad_str, "")
            context_parts.append(f"[TELA DO COMPUTADOR]: {clean_vision}")

        extra_context = "\n".join(context_parts)
        llm.add_user_message(text)

        # Tool-calling loop
        try:
            for i in range(Config.MAX_TOOL_ROUNDS):
                # Passa o extra_context apenas no primeiro round dessa fala
                msg = await llm.get_response(extra_context=extra_context if i == 0 else "")

                # Executar tool calls se houver
                if msg.tool_calls:
                    Logger.ia("Executando ações...")
                    for tool_call in msg.tool_calls:
                        name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        result = await toolbox.execute(name, args)
                        llm.add_tool_result(tool_call.id, name, result)
                    continue  # Próximo round para o LLM processar resultados

                # Resposta final em texto
                if msg.content:
                    Logger.ia(msg.content)
                    await audio.speak(msg.content, interrupt_event)
                break

        except Exception as e:
            Logger.err(f"Erro na inferência: {e}")
            if "connection" in str(e).lower():
                Logger.sys("Conexão perdida. Tentando reconectar...")
                await inf_manager.boot_sequence()

    # ── Tasks Assíncronas ────────────────────────────────────
    async def proactive_loop():
        """Loop que observa a tela e decide se a EVA deve comentar sozinha."""
        await asyncio.sleep(20)  # Espera inicial
        while True:
            import random
            await asyncio.sleep(random.randint(40, 80)) # Tenta a cada ~1 min
            
            if not audio.is_playing and not ptt.is_pressed and vision.current_text:
                thought = await llm.get_proactive_thought(vision.current_text)
                if thought and "[SILÊNCIO]" not in thought.upper():
                    Logger.ia(thought)
                    await audio.speak(thought, interrupt_event)

    tasks = [
        asyncio.create_task(transcription.record_loop()),
        asyncio.create_task(
            transcription.ptt_transcribe_loop(ptt, audio, interrupt_event, handle_command)
        ),
        asyncio.create_task(vision.update_loop()),
        asyncio.create_task(proactive_loop()),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        transcription.stop()
        vision.stop()
        Logger.sys("EVA encerrada.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        Logger.sys("EVA encerrada pelo usuário. Até mais!")
