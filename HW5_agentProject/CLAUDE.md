# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a minimal AI agent project containing `agentSecure.py` - an interactive AI assistant with memory capabilities that can execute shell commands.

## Architecture

**agentSecure.py** - A self-contained AI agent with the following components:

- **Model**: Uses Ollama with `gemma3:27b` model (configurable via `MODEL` constant)
- **Memory System**: 
  - `conversation_history` - stores recent conversation turns (limited to `MAX_TURNS=5`)
  - `key_info` - extracts and stores important information for long-term memory
  - Context is built dynamically and injected into prompts
- **Tool Use**: Parses `<shell>...</shell>` XML tags from AI responses to execute commands
- **Security**: Path safety checking via `is_path_safe()` - blocks external file access unless approved
- **Interaction Loop**: 
  1. Receive user input
  2. Build context from memory/history
  3. Call Ollama API for response
  4. Execute any shell commands found in response (with path safety checks)
  5. Feed results back to AI until `<end/>` is reached
  6. Update memory with conversation

## Running the Agent

```bash
# Start the agent
python agentSecure.py

# Commands within the agent
/quit, /exit, /q  - Exit the agent
/memory           - Display stored key information
```

## Dependencies

- Python 3.x
- `aiohttp` - for async HTTP requests to Ollama
- `ollama` running locally on `localhost:11434`

Install Python dependencies:
```bash
pip install aiohttp
```

## Security: Path Safety Checking

The `is_path_safe()` function intercepts file access commands:

- Extracts paths from commands using `extract_paths_from_command()`
- Normalizes paths with `os.path.abspath()` and `os.path.expanduser()`
- Blocks paths outside `WORKSPACE` (~/.agentSecure) unless explicitly approved
- Detects `..` and `~/` traversal attempts
- Allows system commands in `/usr/`, `/bin/`, `/sbin/`, `/lib/`, `/etc/`

Commands requiring path checks: `cat`, `rm`, `cp`, `mv`, `ln`, `chmod`, `touch`, `mkdir`, `ls`, `find`, `grep`, `python3`, etc.

## Extension Points

- **System Prompt**: Modify `SYSTEM_PROMPT` constant to change agent behavior
- **Model**: Change `MODEL` constant to use different Ollama models
- **Memory Size**: Adjust `MAX_TURNS` to control conversation history depth
- **Workspace**: `WORKSPACE` variable configures the agent's working directory

## API Endpoints Used

- Ollama: `http://localhost:11434/api/generate` (model must be pulled first)
