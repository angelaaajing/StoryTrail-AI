# StoryTrail-AI

## Introduction

**StoryTrail AI** is an intelligent multi-modal travel creation assistant that helps you weave your journey into captivating stories. It integrates photos, videos, text, and audio to automatically generate rich travel narratives, powered by state-of-the-art AI models such as LangChain and Retrieval-Augmented Generation (RAG).

- Supports multi-modal inputs: images, videos, text, and audio  
- Uses local vector databases (FAISS/Chroma) for fast, privacy-preserving retrieval  
- Built on LangChain with modular agent architecture for flexible interaction  
- Easy-to-use Gradio interface for interactive travel story creation  
- Containerized deployment with Docker for reproducibility and scalability  

## Slogan

*“Weave your journey into stories — the intelligent multi-modal travel creator.”*

## Features

- Multi-modal content upload and embedding extraction  
- Retrieval-augmented story generation with Large Language Models (LLMs)  
- Local-first design ensuring user data privacy  
- Interactive agent interface for creative storytelling  

## Getting Started

### Prerequisites

- Python 3.10+  
- Docker (optional, for containerized deployment)  
- Redis (for caching, included in docker-compose)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/storytrail-ai.git
cd storytrail-ai
```

2. Run setup and start scripts:
```bash
bash scripts/setup_env.sh
bash scripts/start_app.sh
```

Or setup manually:

```bash
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
docker run -d -p 6379:6379 --name redis redis:7-alpine
python app/main.py
```

4. Use Docker Compose:

```bash
docker-compose up --build
```

## Project Structure

```
storytrail-ai/
├── app/                   # Core application code (Gradio UI, Agent logic)
├── models/                # Model weights and fine-tuned checkpoints
├── data/                  # Sample travel media and test data
├── scripts/               # Utility scripts (video processing, embedding extraction)
├── tests/                 # Unit and integration tests
├── Dockerfile             # Docker build instructions
├── docker-compose.yml     # Multi-container orchestration (Redis, app)
├── requirements.txt       # Python dependencies
└── README.md              # This document
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)