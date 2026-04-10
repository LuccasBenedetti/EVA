# EVA 4.0 — Assistente Virtual Modular

EVA é uma assistente virtual focada em execução local, baixa latência e integração com o sistema. O diferencial aqui é o uso de **Push-to-Talk (PTT)** para evitar acionamentos acidentais e um loop de **visão proativa** que permite que ela comente o que está acontecendo na sua tela de tempos em tempos.

## Estrutura do Projeto

- `eva.py`: Orquestrador principal.
- `modules/`:
  - `llm.py`: Interface com o vLLM (rodando via WSL2).
  - `transcription.py`: STT usando `faster-whisper`.
  - `vision.py`: Captura de tela e OCR para dar contexto visual à IA.
  - `audio.py`: Gerenciamento de voz (Edge-TTS) e PTT.
  - `memory.py`: Persistência de contexto local.
  - `tools.py`: Funções que a EVA pode executar (automação, busca, etc).

## Requisitos e Setup

### 1. Ambiente Python
Recomendado Python 3.10 ou superior.
```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

### 2. Infraestrutura (vLLM)
A EVA espera um servidor vLLM rodando (preferencialmente no WSL2 para performance de GPU).
- Configure o seu `.env` com a distro correta e o caminho do modelo.
- Ela tentará dar boot no vLLM automaticamente se configurado.

### 3. Configuração (.env)
Copie as variáveis do arquivo `.env` e ajuste conforme necessário:
- `PUSH_TO_TALK_KEY`: Tecla para falar (padrão `f2`).
- `VOICE`: Voz do Edge-TTS.
- `VLLM_MODEL_PATH`: Modelo que será carregado.

## Como Usar

Para iniciar:
```bash
python eva.py
```

### Atalhos e Comandos
- **Segurar F2**: Ativa o microfone. Fale enquanto segura.
- **Soltar F2**: Envia o áudio para transcrição e processamento.
- **Interrupção**: Se você apertar F2 enquanto ela estiver falando, ela para de falar imediatamente para te ouvir.

---

### Notas de Desenvolvimento
- A visão proativa tem um delay configurado no `.env` para não sobrecarregar a inferência.
- O histórico de conversas é mantido em `eva_memory.json`.
- Para adicionar novas funcionalidades, basta criar um método na classe `ToolBox` em `modules/tools.py`.
