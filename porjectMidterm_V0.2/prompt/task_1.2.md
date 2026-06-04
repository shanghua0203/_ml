# Role & Objective
You are a Senior Full-Stack AI Engineer. Your objective is to build a Web-based Chat Interface for an existing custom PyTorch LSTM Language Model using a Backend/Frontend separation architecture (Option B).

# Tech Stack
- **Backend**: `FastAPI` (Python)
- **Frontend**: Vanilla HTML / CSS / JavaScript (Using Fetch API for async requests. Keep it in a `static/` or `templates/` folder served by FastAPI to avoid complex Node.js setups).
- **Testing**: `pytest` and `httpx` (for FastAPI TestClient).

# Core Features to Implement
1. **Model Checkpoint Scanner**: An endpoint (e.g., `GET /api/models`) that scans the `checkpointHistory/` directory and returns a list of available `.pt` files.
2. **Inference Engine API**: An endpoint (e.g., `POST /api/generate`) that accepts a JSON payload containing:
   - `start_word` (string)
   - `model_filename` (string)
   - `temperature` (float)
   - `top_k` (int)
   - `max_length` (int)
   And returns the generated text sequence from the PyTorch model.
3. **Web UI**: A two-column layout.
   - **Left Sidebar**: A control panel with a dropdown for model selection, and sliders for Temperature (0.1 to 2.0) and Top-k (1 to 50).
   - **Right Main Area**: A chat-like interface displaying the user's prompt and the model's generated text, with an input box and a submit button.

# Testing Requirements
You must create a `tests/` directory with comprehensive testing:
1. **Unit Tests (`test_unit.py`)**: Test the core Python utility functions (e.g., text tokenization, tensor shape manipulation, top-k filtering logic). Mock the actual PyTorch model inference to ensure fast execution.
2. **System/Integration Tests (`test_system.py`)**: Use `FastAPI TestClient` to test the API endpoints. Verify that `GET /api/models` returns a 200 OK with a list, and `POST /api/generate` correctly processes parameters and returns a valid response schema.

# Documentation & Language Constraints (CRITICAL)
1. **Language**: ALL code comments, docstrings, variable explanations, and UI text MUST be in Traditional Chinese (繁體中文). Do NOT use Simplified Chinese.
2. **README.md**: You must write a highly detailed `README.md` in Traditional Chinese. It must include:
   - 專案簡介 (Project Introduction)
   - 系統架構說明 (Architecture Overview)
   - 環境建置與套件安裝 (Setup & Installation instructions, including virtual environment)
   - 如何啟動伺服器 (How to start the FastAPI server via uvicorn)
   - 如何執行測試 (How to run the pytest suite)

# Execution
Please output the exact directory structure you plan to create, followed by the complete code for the FastAPI backend (`main_web.py` or `app.py`), the HTML/JS frontend (`index.html`), the testing scripts, and the detailed `README.md`.