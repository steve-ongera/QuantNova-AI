# QuantNova AI 

> A self-hosted, AI-powered Forex analysis platform built with Django + React. No external AI APIs — everything runs locally.

---

## Overview

QuantNova AI is a full-stack forex trading intelligence platform that uses locally-trained machine learning models to:

- Analyze forex chart screenshots (candlestick pattern detection)
- Generate Buy / Sell / Hold predictions with confidence scores
- Extract trading knowledge from PDFs (ICT, SMC, strategies)
- Track trade journal entries and outcomes
- Continuously improve predictions through self-training pipelines

---

## Tech Stack

| Layer       | Technologies                                      |
|-------------|---------------------------------------------------|
| Frontend    | React + Vite, TailwindCSS, Recharts, Axios        |
| Backend     | Django 5, Django REST Framework, Celery, Redis    |
| Database    | PostgreSQL                                        |
| AI/ML       | TensorFlow, PyTorch, Scikit-learn, OpenCV         |
| NLP         | spaCy, Transformers, NLTK                         |
| DevOps      | Docker, Docker Compose, GitHub Actions            |

---

## Project Structure

```
quantnova/
├── backend/                    # Django backend
│   ├── config/                 # Project settings, main URLs
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── celery.py
│   ├── apps/
│   │   ├── users/              # Auth, profiles
│   │   ├── analysis/           # Chart uploads, predictions
│   │   ├── journal/            # Trade journal
│   │   ├── strategies/         # PDF strategy storage
│   │   └── training/           # AI training pipeline management
│   ├── manage.py
│   └── requirements.txt
├── frontend/                   # React + Vite app
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── api/
│   └── package.json
├── ai_engine/                  # Local AI core
│   ├── image_analyzer.py       # OpenCV + CNN chart analysis
│   ├── pattern_detector.py     # Candlestick pattern detection
│   ├── pdf_extractor.py        # PDF → trading knowledge NLP
│   ├── predictor.py            # Main prediction engine
│   └── trainer.py              # Model training scripts
├── datasets/                   # Training data
│   ├── charts/                 # Labeled forex chart images
│   ├── strategies/             # Trading strategy PDFs
│   └── historical/             # CSV historical OHLCV data
├── models_trained/             # Saved trained model files (.h5, .pt)
├── training/                   # Training pipeline configs
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (recommended)

---

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/quantnova-ai.git
cd quantnova-ai
```

---

### 2. Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://postgres:password@localhost:5432/quantnova
REDIS_URL=redis://localhost:6379/0
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:5173
MEDIA_ROOT=media/
```

---

### 3. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

---

### 4. Start Celery Worker (for AI tasks)

```bash
# In a separate terminal, inside backend/
celery -A config worker --loglevel=info
celery -A config beat --loglevel=info
```

---

### 5. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

### 6. Docker (Full Stack)

```bash
docker-compose up --build
```

Services:
- Django API → http://localhost:8000
- React App  → http://localhost:5173
- Redis      → localhost:6379
- PostgreSQL → localhost:5432

---

## AI Engine

### How It Works

```
User uploads chart image
        ↓
OpenCV preprocesses image
        ↓
CNN model detects patterns
        ↓
Pattern → Prediction Engine
        ↓
Buy / Sell / Hold + Confidence %
        ↓
Result stored, journal updated
        ↓
Outcome feedback → Retraining pipeline
```

### Training Your Own Models

```bash
# Place labeled chart images in datasets/charts/
# Format: datasets/charts/buy/chart001.png
#         datasets/charts/sell/chart001.png
#         datasets/charts/hold/chart001.png

cd ai_engine
python trainer.py --epochs 50 --batch-size 32
```

---

## API Endpoints

| Method | Endpoint                        | Description                  |
|--------|---------------------------------|------------------------------|
| POST   | `/api/auth/register/`           | Register new user            |
| POST   | `/api/auth/login/`              | Login, get JWT token         |
| POST   | `/api/analysis/upload/`         | Upload chart for analysis    |
| GET    | `/api/analysis/results/{id}/`   | Get prediction result        |
| GET    | `/api/analysis/history/`        | List past analyses           |
| POST   | `/api/journal/`                 | Create trade journal entry   |
| GET    | `/api/journal/`                 | List journal entries         |
| POST   | `/api/strategies/upload/`       | Upload strategy PDF          |
| GET    | `/api/strategies/`              | List extracted strategies    |
| POST   | `/api/training/trigger/`        | Trigger model retraining     |
| GET    | `/api/training/status/`         | Check training job status    |

---

## Roadmap

### Version 1 — Core AI Analysis  (Current)
- [x] Chart image upload
- [x] AI trend detection
- [x] Buy/Sell/Hold prediction
- [x] Confidence score
- [x] Trade journal

### Version 2 — Advanced Intelligence
- [ ] Backtesting engine
- [ ] Multi-timeframe analysis
- [ ] News sentiment scoring
- [ ] Risk/reward calculator
- [ ] Auto-retraining on outcomes

### Version 3 — Pro Platform
- [ ] Reinforcement learning agent
- [ ] AI trading assistant chat
- [ ] Multi-broker integration
- [ ] Voice analysis
- [ ] Mobile app (React Native)

---

## Realistic AI Expectations

| Metric                         | Realistic Range     |
|--------------------------------|---------------------|
| Pattern detection accuracy     | 65–80%              |
| Buy/Sell signal confidence     | 55–75% on good setups|
| PDF knowledge extraction       | ~85% rule capture   |
| Improvement over time          | Yes, with feedback  |

>  Forex is inherently probabilistic. No system achieves 100% accuracy. QuantNova AI gives you **probabilistic edge**, not certainty.

---

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built with Love  for serious traders who want an edge.

**QuantNova AI** — *Trade with intelligence. Grow with data.*