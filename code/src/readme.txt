╔══════════════════════════════════╗
║AI-Powered Hyper-Personalization Engine
╚══════════════════════════════════╝

TECHNICAL STACK:
- Generative AI: OPT-125M LLM (text generation)
- Embedding AI: Sentence-BERT (semantic analysis)
- Multi-modal: VOSK (voice) + Tesseract (images)
- Framework: FastAPI + Transformers

QUICK START:
1. Install dependencies:
   pip install -r requirements.txt

AI-Driven Hyper-Personalization Engine

===============================
QUICK SETUP
===============================

1. Install requirements:
pip install -r requirements.txt

2. Run server:
python aipersonalize.py

OR (alternative):
uvicorn aipersonalize:app --reload

===============================
API USAGE
===============================

POST /analyze/
- Main endpoint for recommendations
- Required: Excel file (customer data)
- Optional:
  • Audio file (voice queries) - WAV format recommended
  • Image file (receipts/screenshots) - JPG/PNG format

Example curl:
curl -X POST "http://localhost:8000/analyze/" \
  -F "file=@customers.xlsx" \
  -F "voice=@query.wav" \    # Optional
  -F "image=@receipt.jpg"    # Optional

===============================
SUPPORTED FEATURES
===============================
✓ Customer profile analysis
✓ Transaction pattern detection
✓ Voice query processing (optional)
✓ Image text extraction (optional)
✓ Multi-factor recommendations
✓ Bias detection

===============================
NOTES
===============================
- Voice/image processing automatically skips if files not provided
- System requirements: 4GB+ RAM for optimal performance
- First run may take longer to download AI models