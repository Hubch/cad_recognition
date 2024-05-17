# cad detect system

## 🛠 Getting Started

The app has a React/Vite frontend and a FastAPI backend and model serve is use ray serve. 

Run the backend (I use Poetry for package management - `pip install poetry` if you don't have it):

```bash
cd backend
echo "OPENAI_API_KEY=sk-your-key" > .env
poetry install
poetry shell
poetry run uvicorn main:app --reload --port 8001
```

If you want to use Anthropic, add the `ANTHROPIC_API_KEY` to `backend/.env` with your API key from Anthropic.

Run the frontend:

```bash
cd frontend
yarn
yarn dev
```

Open http://localhost:5173 to use the app.


If you prefer to run the backend on a different port, update VITE_WS_BACKEND_URL in `frontend/.env.local`

Run model serve
```bash
cd modelserve
pip install -r requirements.txt
ray start --head
serve run ./config.yaml
```



## Docker

If you have Docker installed on your system, in the root directory, run:

```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
docker-compose up -d --build
```

The app will be up and running at http://localhost:5173. Note that you can't develop the application with this setup as the file changes won't trigger a rebuild.

## 🙋‍♂️ FAQs


