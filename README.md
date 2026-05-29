---
title: RAG PDF Assistant
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
short_description: AI assistant that answers questions from PDF documents
---

# 📄 RAG PDF Assistant

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/laffont-dev/rag-pdf-assistant-demo)

Un assistant conversationnel RAG (Retrieval-Augmented Generation) qui permet d'interroger vos documents PDF en langage naturel.

Posez des questions sur n'importe quel document PDF — l'assistant extrait les passages pertinents et génère une réponse précise, sourcée et sans hallucination.

---

## 🚀 Démo live

👉 **[rag-pdf-assistant-demo](https://huggingface.co/spaces/laffont-dev/rag-pdf-assistant-demo)**

Testez l'application avec un PDF d'exemple ou importez vos propres documents.

---

## 💡 Cas d'usage

- **RH & onboarding** — Interroger le manuel de l'employé, la politique interne
- **Documentation technique** — Poser des questions sur une documentation PDF
- **Contrats & juridique** — Chercher des clauses spécifiques dans un contrat
- **Éducation & recherche** — Explorer des articles académiques ou des notes de cours
- **Support client** — Accéder rapidement aux procédures depuis un manuel

---

## ⚙️ Stack technique

| Composant        | Technologie                                           |
| ---------------- | ----------------------------------------------------- |
| Interface        | [Streamlit](https://streamlit.io)                     |
| PDF parsing      | [PyMuPDF](https://pymupdf.readthedocs.io) (fitz)      |
| Embeddings       | [sentence-transformers](https://www.sbert.net) (all-MiniLM-L6-v2) |
| Vector search    | [FAISS](https://faiss.ai) (IndexFlatL2)               |
| LLM              | [Mistral API](https://mistral.ai) (mistral-small-latest) |
| Langage          | Python 3.10+                                          |

---

## 🛠️ Lancer localement

```bash
# 1. Cloner le dépôt
git clone https://github.com/laffont-dev/rag-pdf-assistant-demo.git
cd rag-pdf-assistant-demo

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé API Mistral
cp .env.example .env
# Éditer .env et ajouter votre clé : MISTRAL_API_KEY=***

# 4. Placer un PDF d'exemple dans data/
# (ou utiliser l'upload dans l'interface)

# 5. Lancer l'application
streamlit run app.py
```

---

## 📦 Architecture

```
rag-pdf-assistant-demo/
├── app.py                  # Application Streamlit
├── rag/
│   ├── __init__.py         # Version du package
│   ├── pdf_loader.py       # Chargement PDF avec PyMuPDF
│   ├── chunker.py          # Découpage en chunks avec overlap
│   ├── embedder.py         # Embeddings avec sentence-transformers
│   ├── vector_store.py     # Index vectoriel FAISS
│   └── llm.py              # Appel API Mistral
├── data/                   # PDFs (manuel d'exemple, uploads temporaires)
├── assets/                 # Ressources statiques
├── tests/                  # Tests unitaires et d'intégration
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_rag_pipeline.py
├── .env.example            # Template de configuration
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
```

---

## 🔧 Déploiement HuggingFace Spaces

1. Créez un Space sur [huggingface.co/spaces](https://huggingface.co/spaces) avec SDK **Streamlit**
2. Poussez les fichiers du dépôt dans le Space
3. Ajoutez vos secrets (Settings → Repository secrets) :
   - `MISTRAL_API_KEY` : votre clé API Mistral
4. Le Space se rebuild automatiquement
5. Le premier démarrage télécharge le modèle `all-MiniLM-L6-v2` (~80 Mo)

> ⚠️ Sur le plan gratuit (CPU basic), la réponse peut prendre 5-10 secondes au premier démarrage (téléchargement du modèle).

---

## 🔑 Variables d'environnement

| Variable           | Obligatoire | Description                              |
| ------------------ | ----------- | ---------------------------------------- |
| `MISTRAL_API_KEY`  | ✅          | Clé API Mistral (console.mistral.ai)     |

---

## 📩 Contact

**hello@laffont.dev** — Disponible pour missions freelance

---

*Aurélien Laffont — Freelance IA & Automatisation*