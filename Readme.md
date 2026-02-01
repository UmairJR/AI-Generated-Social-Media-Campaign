# AI-Generated Social Media Campaign Evasion

This project demonstrates how **AI-generated social media campaigns** can evade
traditional **text-similarity-based detection algorithms** while preserving
coordinated behavior.

Inspired by:
- Pohl et al. (2022) – Artificial Social Media Campaign Creation
- Assenmacher et al. (2021) – Benchmarking Crisis Detection

---

## Objective

- Detect coordinated social media campaigns
- Extract campaign structure (blueprint)
- Rewrite campaign tweets using AI
- Show how detection fails after AI rewriting
- Compare lexical vs semantic clustering methods

---

## Technologies

- Python 3.9+
- Streamlit
- OpenAI API
- scikit-learn
- sentence-transformers (SBERT)
- HDBSCAN
- pandas, numpy, matplotlib

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd ai-campaign-evasion
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. OpenAI API Setup
```bash
setx OPENAI_API_KEY "your_api_key_here"
```
### 4. Run Application
```bash
streamlit run app.py
```
### 5. Other settings
 - Keep threshold = 0.5
 - SBERT+HDBSCAN is optional (Trying semantic based algorithm for results)
 - Only textclust was the requirement
---

### Developers
- Umair Shaikh
- Hamid
- Arhaan Khan
