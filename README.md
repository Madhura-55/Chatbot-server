# Chatbot-server

```text
Chatbot-server/
├── .env.example
├── .gitignore
├── requirements.txt
├── config.py
├── main.py
├── Dockerfile
├── docker-compose.yml
├── services/
│   ├── __init__.py
│   ├── mongo_service.py
│   ├── vector_store.py
│   ├── gemini_service.py
│   └── rag_pipeline.py
├── data/
│   └── policies/
│       ├── return_policy.md
│       ├── shipping_policy.md
│       └── faq.md
├── scripts/
│   └── ingest_policies.py
└── widget/
    └── chatbot-widget.js
```
