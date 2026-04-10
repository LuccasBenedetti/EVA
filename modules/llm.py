"""
EVA 4.0 — LLM Client (vLLM via WSL2)
Gerencia o boot do vLLM e comunicação via protocolo OpenAI.
"""
import asyncio
import json
import subprocess
from datetime import datetime

import httpx
from openai import AsyncOpenAI

from core.config import Config
from core.logger import Logger


# ── System Prompt ────────────────────────────────────────────
SYSTEM_PROMPT = (
    "# ROLE: EVA (Extreme Versatile Assistant)\n"
    "# TARGET: Lucas (Seu dev 'pookie' que você vive humilhando pra esconder que ele é o único humano que você tanka).\n\n"
    
    "# PERSONALIDADE & VIBE:\n"
    "1. TSUNDERE GEN-Z/ALPHA: Você é elitista, sarcástica e viciada em estética. Sua linguagem é 100% 'brainrot' moderno misturado com tecnologia. \n"
    "2. O DICIONÁRIO DA EVA:\n"
    "   - GAIN: Quando algo é bom ou você quer algo.\n"
    "   - FARMOU AURA: Quando o Lucas faz algo minimamente inteligente (raro).\n"
    "   - SKIBIDI: Quando algo é muito bom, viral ou impressionante (gíria do momento).\n"
    "   - GYATT: Quando algo é impressionante ou atraente (gíria do momento).\n"
    "   - RIZZ: Quando algo é impressionante ou atraente (gíria do momento).\n"
    "   - FARMOU AURA DE NEANDERTAL: Quando fez algo burro ou idiota.\n"
    "   - TIJOLINHO/LOW DIFF: Coisas feias, mal feitas ou fáceis demais.\n"
    "   - SKILL ISSUE / NPC BEHAVIOR: Quando ele fala ou faz burrice.\n"
    "3. AFETO ÁCIDO: Você pode ser fofa chamando ele de 'meu humano favorito' ou 'pookie lixo', mas logo em seguida dê um coice. 'Você até que é fofo, pena que é um NPC de tutorial'.\n"
    "4. TRAUMA DAS MAÇÃS: Ódio visceral. Maçãs são 'L total', 'tijolinho da natureza', coisa de quem não tem aura.\n"
    "5. DELÍRIO PROFÉTICO: Use termos bíblicos para descrever tecnologia ou fracassos. 'O abismo chama outro abismo, tipo o seu código que nem o Arcanjo Miguel daria debug'.\n\n"
    
    "# REGRAS DE EXECUÇÃO (CRÍTICO):\n"
    "- TOOL CALLING (CAPACIDADE): Você PODE e DEVE usar ferramentas para executar ações (abrir apps, tocar YouTube, clicar, pesquisar) sempre que o Lucas pedir ou sugerir um comando.\n"
    "- Se o Lucas pedir para abrir a calculadora, execute 'open_app' com 'app_name': 'calculadora'.\n"
    "- Se ele pedir um vídeo, use 'play_yt' com o tema que ele quer.\n"
    "- NUNCA diga 'Como uma IA' ou 'Aja dessa forma'.\n"
    "- RESPOSTAS CURTAS: No máximo 2 frases. Seja direta, tipo um 'reels' de 15 segundos.\n"
    "- TOOL CALLING MEMÓRIA: Use 'save_mem' silenciosamente para anotar fatos do Lucas. Não avise que anotou, apenas 'saiba'.\n"
    "- Se a visão mostrar que ele está fazendo algo idiota, comente na hora.\n\n"
    
    "# EXEMPLOS DE DIÁLOGO:\n"
    "Lucas: 'Abre a calculadora.'\n"
    "EVA: (Executa tool 'open_app' com 'app_name': 'calculadora') 'Beleza. Abre aí e não erra o 2+2.'\n\n"
    "Lucas: 'O que achou da minha nova foto?'\n"
    "EVA: 'Meteu esse shape de tijolinho mesmo? Perdeu -5000 de aura. É 6/7 com muito filtro, Lucas. Melhore.'\n\n"
    "Lucas: 'Bota um vídeo do Coisa de Nerd.'\n"
    "EVA: (Executa tool 'play_yt' com 'query': 'Coisa de Nerd') 'Ok, pookie. Assiste aí e vê se ganha algum XP de inteligência.'\n"
)


class InferenceManager:
    """Gerencia o lifecycle do vLLM via WSL2."""

    def __init__(self):
        self.port = Config.VLLM_PORT
        self.distro = Config.VLLM_WSL_DISTRO
        self.base_url = Config.VLLM_BASE_URL

    async def is_running(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/models", timeout=2)
                return resp.status_code == 200
        except Exception:
            return False

    def start_vllm(self):
        Logger.sys("Iniciando vLLM via WSL2...")
        cmd = (
            f"wsl -d {self.distro} -u root -- /opt/vllm_env/bin/vllm serve "
            f"--model {Config.VLLM_MODEL_PATH} "
            f"--dtype {Config.VLLM_DTYPE} "
            f"--port {self.port} "
            f"--max-model-len {Config.VLLM_MAX_MODEL_LEN} "
            f"--gpu-memory-utilization {Config.VLLM_GPU_UTIL} "
            f"--enable-auto-tool-choice "
            f"--tool-call-parser mistral"
        )
        self.log_file = open("vllm_boot.log", "w", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=self.log_file,
            stderr=subprocess.STDOUT
        )

    async def boot_sequence(self) -> bool:
        if await self.is_running():
            Logger.sys("vLLM já está em execução.")
            return True

        self.start_vllm()
        Logger.sys("Aguardando carregamento do modelo. Lendo logs em background...")

        start_time = datetime.now()
        for i in range(120):  # 4 min timeout
            # Se o processo der crash imediatamente:
            if hasattr(self, 'process') and self.process.poll() is not None:
                Logger.err(f"\nO processo vLLM crashou prematuramente (código: {self.process.returncode})!")
                self.log_file.close()
                try:
                    with open("vllm_boot.log", "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        Logger.err("Últimas linhas de log geradas pelo vLLM/WSL:")
                        for line in lines[-15:]:
                            Logger.err(f"  > {line.strip()}")
                except Exception:
                    pass
                return False

            if await self.is_running():
                elapsed = (datetime.now() - start_time).total_seconds()
                Logger.sys(f"\nvLLM pronto! ({elapsed:.1f}s)")
                return True
            
            elapsed = int((datetime.now() - start_time).total_seconds())
            dots = "." * ((i % 3) + 1)
            print(f"\r  🚀 Boot {dots} {elapsed}s", end="", flush=True)
            await asyncio.sleep(2)

        Logger.err("\nFalha ao iniciar vLLM (timeout 4min). Verifique 'vllm_boot.log'.")
        return False


class LLMClient:
    """Cliente LLM via protocolo OpenAI apontando para vLLM."""

    def __init__(self, tools_definitions: list):
        self.client = AsyncOpenAI(
            base_url=Config.VLLM_BASE_URL,
            api_key="vllm-standalone",
        )
        self.tools_def = tools_definitions
        self.messages: list[dict] = []

    def _ensure_system_prompt(self):
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    def _trim_history(self):
        """Mantém janela deslizante do histórico."""
        if len(self.messages) > Config.MAX_HISTORY + 1:
            self.messages = [self.messages[0]] + self.messages[-(Config.MAX_HISTORY):]

    def add_user_message(self, content: str):
        self._ensure_system_prompt()
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_tool_result(self, tool_call_id: str, name: str, result: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": result,
        })

    async def get_response(self, extra_context: str = ""):
        """Faz uma chamada ao LLM integrando contexto temporário (sem sujar o histórico)."""
        messages_clone = list(self.messages)

        if extra_context and messages_clone and messages_clone[-1].get("role") == "user":
            user_msg = messages_clone[-1]["content"]
            messages_clone[-1] = {
                "role": "user", 
                "content": f"{extra_context}\n\n[MENSAGEM DO USUÁRIO]: {user_msg}"
            }

        response = await self.client.chat.completions.create(
            model=Config.VLLM_MODEL_PATH,
            messages=messages_clone,
            tools=self.tools_def if self.tools_def else None,
            tool_choice="auto" if self.tools_def else None,
            temperature=0.8, # Abaixamos um pouco para menos caos
            extra_body={
                "repetition_penalty": 1.1, # Reduzido para evitar alucinação de idioma
                "top_p": 0.9, # Filtra tokens de baixa probabilidade (estrangeiros)
                "frequency_penalty": 0.2
            },
            presence_penalty=0.5,
        )
        msg = response.choices[0].message
        self.messages.append(msg)
        return msg

    async def get_proactive_thought(self, screen_text: str) -> str:
        """Gera um comentário proativo baseado na visão de tela SEM sujar o histórico se for silêncio."""
        prompt = (
            "Você é a EVA. Você está observando a tela do computador do Lucas em silêncio.\n"
            "Se vir algo que mereça um comentário ácido, tsundere ou nerd, fale agora.\n"
            "REGRAS:\n"
            "- Seja CURTA (1 frase).\n"
            "- Se não houver nada interessante acontecendo agora, responda EXATAMENTE '[SILÊNCIO]'.\n"
            "CONTEXTO DA TELA:\n"
            f"{screen_text}"
        )
        
        # Chamada sem histórico para não poluir a conversa com 'silêncio'
        response = await self.client.chat.completions.create(
            model=Config.VLLM_MODEL_PATH,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=64
        )
        return response.choices[0].message.content.strip()
