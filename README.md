> The model is a small experimental local language model. It can generate useful structured suggestions, but a larger and more diverse dataset is the most important next step for improving accuracy.

## Project Structure

```text
domain-suggesion-llm/
├── backend-models/
│   ├── app/              # FastAPI application and API endpoints
│   ├── checkpoints/      # Local trained model checkpoints (ignored by Git)
│   ├── data/             # Raw, filtered, and cleaned training data
│   ├── inference/        # Checkpoint loading and text generation
│   ├── model/            # Decoder-only Transformer implementation
│   ├── tokenizer/        # BPE tokenizer files and training script
│   └── training/         # Dataset construction and training loop
└── frontend/
    └── app/              # Next.js chat user interface and API proxy
```

## Run the Local AI Application

### 1. Start the Transformer API

From the project root:

```powershell
cd backend-models
..\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
```

Verify the API is live:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "domain-transformer"
}
```

### 2. Start the Next.js AI Chat UI

Open a second terminal from the project root:

```powershell
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in a browser.

## API Reference

### `POST /api/suggest`

The FastAPI service accepts a project or skills description and returns a generated suggestion.

```json
{
  "message": "I build React applications using TypeScript and Tailwind CSS."
}
```

Example response:

```json
{
  "reply": "Domain: Web Development\nProfession: Frontend Engineer\nReason: ...\nSkills: React, TypeScript",
  "model": "domain-transformer",
  "generation_ms": 389
}
```

The browser talks to `frontend/app/api/chat`, which proxies the request to the local Python API at `http://127.0.0.1:8000`.

## Train the Transformer Model

The tokenizer and current best checkpoint are already created. To start another training run:

```powershell
cd backend-models
..\.venv\Scripts\python.exe -m training.pretrain --epochs 50 --batch-size 8
```

The training process writes `best.pt` and `last.pt` to `backend-models/checkpoints/pretrain/`.

## Technology Stack

- **PyTorch** – Transformer architecture and model training
- **Tokenizers** – Byte-Level BPE tokenizer training and decoding
- **FastAPI + Uvicorn** – Local Python model inference API
- **Next.js 16 + React 19** – AI chat interface and API proxy
- **Tailwind CSS** – Chat panel styling

## Future Improvements

- Increase the domain-suggestion dataset beyond 106 examples
- Add structured output validation and retry logic
- Improve model evaluation with held-out domain prompts
- Add streaming generation to the chat UI
- Add a dedicated model card with dataset and training lineage
- Deploy the frontend and model API with secure configuration

## License

This project is currently intended for local development and experimentation.
