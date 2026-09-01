# Natural Language Processing with Generative AI: Medical Assistant

<!-- Live Demo / Notebook Badges -->
<!--  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-medical-assistant.streamlit.app/) -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://ashish1100.github.io/Natural-Language-Processing-with-Generative-AI-Medical-Assistant/)
[![License: Non-Commercial](https://img.shields.io/badge/License-Educational%20%2F%20Non--Commercial-green.svg)](https://github.com/Ashish1100/Natural-Language-Processing-with-Generative-AI-Medical-Assistant/blob/c65cfc8d117f1e4c903ad828782615e61b40edb3/license.md)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ashish-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/ashishsaha21/)

<!-- Core Technology Stack Badges -->
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Mistral AI](https://img.shields.io/badge/Mistral-7B--Instruct--v0.2-FF7000?style=flat&logo=mistralai&logoColor=white)](https://mistral.ai/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-GPU%20cuBLAS-000000?style=flat)](https://github.com/ggerganov/llama.cpp)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F61?style=flat)](https://www.trychroma.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-blue?style=flat)](https://www.sbert.net/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hub-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF%20Ingestion-2B579A?style=flat)](https://pymupdf.readthedocs.io/)

<!-- License & Social Badges -->


<!--## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [System Architecture](#system-architecture)
5. [Technology Stack](#technology-stack)
6. [Pipeline Workflow](#pipeline-workflow)
7. [Data Description](#data-description)
8. [Implementation Details](#implementation-details)
9. [Evaluation Methodology](#evaluation-methodology)
10. [Results and Analysis](#results-and-analysis)
11. [Business Impact and Recommendations](#business-impact-and-recommendations)
12. [Getting Started](#getting-started)
13. [Project Structure](#project-structure)
14. [Author](#author) --> 

---

## Project Overview

Medical Assistant is a Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-aware responses to medical queries by leveraging the comprehensive Merck Manuals. The system addresses the critical challenge of information overload in healthcare by enabling rapid access to verified medical knowledge through natural language queries.

The project implements a progressive AI approach, advancing from direct LLM inference through prompt engineering techniques to a fully functional RAG pipeline with automated evaluation.

---

## Problem Statement

Healthcare professionals face increasing challenges in managing vast volumes of medical data while delivering accurate and timely diagnoses. Key challenges include:

- **Information Overload**: Difficulty in sifting through extensive research and medical literature
- **Time-Sensitive Decisions**: Critical need for efficiency in emergency situations
- **Access to Trusted Information**: Requirement for current, reliable medical references
- **Standardization of Care**: Need for consistent treatment protocols across healthcare providers

---

## Objectives

The primary objective is to develop a RAG-based AI solution that:

1. **Understands** healthcare challenges including information overload and decision-making bottlenecks
2. **Applies** AI techniques to streamline access to medical knowledge
3. **Analyzes** the impact on diagnostic accuracy and patient outcomes
4. **Evaluates** potential for standardizing care practices
5. **Creates** a functional prototype demonstrating feasibility and effectiveness

### Target Queries

| Category | Sample Question |
|----------|----------------|
| Critical Care | What is the protocol for managing sepsis in a critical care unit? |
| Surgical | What are the common symptoms for appendicitis, and can it be cured via medicine? |
| Dermatology | What are the effective treatments for sudden patchy hair loss? |
| Neurology | What treatments are recommended for brain tissue injury? |
| Orthopedics | What are the necessary precautions for a leg fracture during hiking? |

---

## System Architecture

<p align="center">
  <img src="./assets/System Architecture.png" alt="NLP Project" width="800"/>
</p>

---

## Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Programming Language** | Python 3.10 | Core development language |
| **LLM Loading and Inference** | Llama-cpp-python | Local GPU-accelerated model inference |
| **Base LLM** | Mistral-7B-Instruct-v0.2 (GGUF, Q6_K) | Medical knowledge and reasoning |
| **Orchestration Framework** | LangChain (0.3.x) | RAG pipeline component chaining |
| **Vector Database** | ChromaDB (1.1.x) | Persistent embedding storage and retrieval |
| **Embedding Model** | Sentence Transformers | Text-to-vector conversion |
| **Embedding Architecture** | all-MiniLM-L6-v2 (384-dimensional) | Dense semantic embeddings |
| **Data Ingestion** | PyMuPDF | PDF document parsing |
| **Text Processing** | RecursiveCharacterTextSplitter | Context-aware chunking |
| **Tokenization** | Tiktoken | Precise token counting |
| **Data Manipulation** | Pandas, NumPy | Tabular processing and evaluation |
| **Model Repository** | Hugging Face Hub | Model download and versioning |
| **Environment** | Jupyter Notebook / Google Colab | Development and execution |

---

## Pipeline Workflow

### Stage 1: Data Ingestion

1. **Document Load**: The Merck Manual PDF (4,114 pages) is loaded using `PyMuPDFLoader`
2. **Token-Aware Chunking**: Documents are split into 512-token chunks with 50-token overlap using `RecursiveCharacterTextSplitter`
3. **Quality Filtering**: Chunks smaller than 20 tokens are removed to maintain retrieval quality

### Stage 2: Embedding and Storage

1. **Embedding Generation**: Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`
2. **Batch Processing**: Chunks are embedded in batches of 100 to manage memory usage
3. **Vector Storage**: Embeddings are persisted in a ChromaDB collection for efficient retrieval

### Stage 3: Retrieval and Generation

1. **Query Processing**: User query is embedded using the same sentence transformer model
2. **Similarity Search**: Top-k (k=3) most relevant chunks are retrieved via cosine similarity
3. **Context Construction**: Retrieved chunks are combined into a single context block
4. **Response Generation**: Mistral-7B-Instruct generates a response based on the retrieved context and system prompts

---

## Data Description

The **Merck Manual of Diagnosis and Therapy, 19th Edition** serves as the knowledge base:

| Attribute | Value |
|-----------|-------|
| **Source** | Merck & Co. medical reference |
| **Format** | PDF |
| **Total Pages** | 4,114 |
| **Sections** | 23 comprehensive medical sections |
| **Chunks Created** | 8,706 |
| **Chunk Size** | 512 tokens |
| **Chunk Overlap** | 50 tokens |

---

## Implementation Details

### Prompt Engineering Approaches

The project demonstrates three progressive approaches to medical query answering:

#### 1. Direct LLM Inference
Standard LLM response generation without additional context, leveraging the model's inherent knowledge.

#### 2. Prompt Engineering
Custom system prompts and parameter tuning to improve response quality:

| Strategy | Use Case | Implementation |
|----------|----------|----------------|
| **Strict Persona** | Core RAG Setup | Professional medical assistant constraints |
| **Creative Persona** | Appendicitis Query | Shakespearean language with medical accuracy |
| **Few-Shot Prompting** | Brain Injury Query | Examples of structured medical responses |
| **Chain-of-Thought** | Leg Fracture Query | Step-by-step reasoning framework |

#### 3. RAG Pipeline
Full retrieval-augmented generation with strict context grounding.

### System Prompts

The system implements strict hallucination guardrails:

```text
You are a highly skilled and knowledgeable medical assistant.
Your task is to provide accurate, concise, and professional medical information
based strictly on the provided context from the Merck Manuals.

Guidelines:
1. Use ONLY the provided context to answer the question.
2. If the context does not contain the answer, state:
   'Based on the provided medical manuals, I do not have sufficient
   information to answer this query.'
3. Maintain a clinical and helpful tone.
4. Ensure instructions or protocols are listed clearly using bullet points.
```

### LLM Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Context Window** | 2,300 tokens | Balances memory and scope |
| **GPU Layers** | 8 | Partial GPU acceleration |
| **Batch Size** | 128 | Optimized throughput |
| **Temperature** | 0 (default) | Deterministic output for medical queries |
| **Top-P** | 0.95 | Controlled sampling |
| **Top-K** | 50 | Vocabulary restriction |

---

## Evaluation Methodology

### LLM-as-a-Judge Framework

The system implements automated evaluation using two metrics with a 1-5 scoring scale:

#### Groundedness

| Score | Criterion |
|-------|-----------|
| 1 | Major clinical errors or contradicts the context |
| 2 | Most claims are not found in the context |
| 3 | Some claims are supported, minor hallucinations |
| 4 | All claims supported, but missing nuances |
| 5 | Every factual claim explicitly supported by context |

#### Relevance

| Score | Criterion |
|-------|-----------|
| 1 | Completely irrelevant or ignores the question |
| 2 | Misses the core medical concern |
| 3 | Partially relevant, lacks specific details |
| 4 | Relevant and addresses main points |
| 5 | Perfectly aligned and provides complete answer |

### Evaluation Pipeline

The evaluation function executes three sequential LLM calls:

1. **Response Generation**: Generates the medical answer using retrieved context
2. **Groundedness Scoring**: Evaluates factual accuracy against provided context
3. **Relevance Scoring**: Evaluates alignment with the user's query

---

## Results and Analysis

### Query Performance Summary

| Query Topic | Groundedness | Relevance | Key Observation |
|-------------|--------------|-----------|-----------------|
| **Sepsis Protocol** | 5 | 4 | Accurate context summarization, minor detail gaps |
| **Appendicitis** | 1 | 4 | Context truncation caused factual grounding loss |
| **Patchy Hair Loss** | 5 | 4 | Correctly identified Alopecia Areata and treatments |
| **Brain Injury** | 4 | 4 | Maintained accuracy despite aggressive truncation |
| **Leg Fracture** | 5 | 5 | Perfect alignment and complete answer |

### Key Findings

#### Technical Insights

1. **Context Window Management**: Aggressive truncation of retrieved chunks significantly degrades Groundedness scores. The Appendicitis query scored 1/5 for Groundedness due to context loss when truncating to fit the 2,300-token window.

2. **Prompt Engineering Effectiveness**: Chain-of-Thought prompting successfully generated chronological, structured responses for complex scenarios like the leg fracture query.

3. **Automated Evaluation Viability**: The LLM-as-a-judge framework successfully identified data loss (Appendicitis: Groundedness 1) and validated high-quality responses (Leg Fracture: 5/5).

4. **Token Parameter Requirements**: Default `max_tokens=128` is insufficient for comprehensive medical answers, causing frequent output truncation.

#### Business Insights

1. **Clinical Decision Support**: Position as an augmentation tool for physicians, not a replacement.

2. **Burnout Reduction**: Reduces time spent on literature search in critical care settings.

3. **Care Standardization**: Anchors responses to vetted institutional knowledge, ensuring consistent treatment protocols.

4. **Privacy Compliance**: Local LLM architecture ensures data privacy for HIPAA/GDPR compliance.

### Hyperparameter Tuning Progress

A grid search was initiated to optimize generation parameters:

| Config | Temperature | Top-P | Top-K | Retriever k | Max Tokens | Status |
|--------|-------------|-------|-------|-------------|------------|--------|
| 1 | 0.0 | 0.90 | 40 | 2 | 256 | Completed |
| 2 | 0.7 | 0.95 | 50 | 2 | 256 | Interrupted |
| 3 | 0.9 | 0.85 | 20 | 1 | 200 | Pending |
| 4 | 0.3 | 1.00 | 60 | 3 | 300 | Pending |
| 5 | 0.5 | 0.90 | 30 | 3 | 256 | Pending |

*Note: Execution was interrupted due to computational resource constraints on the free tier of Google Colab.*

---

## Business Impact and Recommendations

### Strategic Recommendations

1. **Deployment Scenarios**
   - Critical care units for rapid sepsis and emergency protocols
   - Emergency departments for triage support
   - Primary care settings for diagnostic assistance

2. **Technology Enhancements**
   - Implement Parent Document Retrieval to maximize context density
   - Adopt context compression techniques to reduce token usage
   - Integrate real-time Electronic Health Record (EHR) data for personalized responses

3. **Operational Guidelines**
   - Position as a "second opinion" engine for physicians
   - Implement human-in-the-loop validation for high-stakes queries
   - Maintain strict version control for medical knowledge updates

### Compliance Considerations

- **Data Privacy**: Local LLM inference ensures patient data remains on-premises
- **Regulatory Alignment**: Supports HIPAA, GDPR, and DISHA compliance frameworks
- **Clinical Governance**: Requires physician oversight for treatment recommendations

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Google Colab (recommended) or local Jupyter Notebook
- GPU (optional but recommended for faster inference)

### Installation

1. Clone or open the notebook in Google Colab.

2. Install dependencies:

```bash
# Install llama-cpp-python with GPU support
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir -q

# Install core libraries
!pip install huggingface_hub==0.35.3 pandas==2.2.2 tiktoken==0.12.0 pymupdf==1.26.5 langchain==0.3.27 langchain-community==0.3.31 chromadb==1.1.1 sentence-transformers==5.1.1 numpy==2.3.3 -q
```

3. Restart the runtime after installation.

4. Prepare the Merck Manual PDF and place it at `/content/medical_diagnosis_manual.pdf`.

### Usage

```python
# Initialize the RAG pipeline (after running all notebook cells)
query = "What is the protocol for managing sepsis in a critical care unit?"
response = generate_rag_response(query)
print(response)
```

---

## Project Structure

```
Project_Natural_Language_Processing_with_Generative_AI__Medical_Assistant_v2.ipynb
│
├── Section 1: Problem Statement and Business Context
├── Section 2: Library Installation and Configuration
├── Section 3: Question Answering using LLM (Direct)
│   ├── Model Download and Loading
│   └── Five Query Evaluations
├── Section 4: Prompt Engineering and LLM Parameter Tuning
│   ├── Strict Persona Configuration
│   ├── Creative Persona Configuration
│   ├── Zero-Shot and Top-K Tuning
│   ├── Few-Shot Prompting
│   └── Chain-of-Thought Prompting
├── Section 5: Data Preparation for RAG
│   ├── Document Loading
│   ├── Data Overview and Validation
│   ├── Text Chunking (8,706 chunks)
│   ├── Embedding Generation (384-dim vectors)
│   ├── Vector Database Creation (ChromaDB)
│   └── Retriever Configuration
├── Section 6: Question Answering using RAG
│   └── Five Query Evaluations with Retrieved Context
├── Section 7: Hyperparameter Tuning
│   └── Multi-Configuration Grid Search
└── Section 8: Output Evaluation
    ├── Groundedness Scoring
    ├── Relevance Scoring
    └── LLM-as-a-Judge Framework
```
---

## **License & Legal**

```
© 2026 Ashish Saha

This project is a personal initiative intended for educational use only.

Permission is granted to use, copy, and modify this software for learning and research purposes.
Commercial use, sale, or monetization of this software or its derivatives is strictly prohibited.

The software is provided “as is”, without warranty of any kind.

```

---

## Author

<div align="center">

### **Ashish Saha**
**AI Engineering** | **ML Research** | **Data Science**

*Specializing in building intelligent ML systems and transforming data into actionable insights.*

**Tech Stack:** Python • TensorFlow/Keras • PyTorch • XGBoost • Scikit-learn 

<a href="https://github.com/Ashish1100" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub">
</a>
<a href="https://www.linkedin.com/in/ashishsaha21/" target="_blank">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn">
</a>
<a href="mailto:ashishsaha.software@gmail.com">
  <img src="https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Email">
</a>

</div>

---

</div>

<div align="center">

### **Star ⭐ this repo if you found this project helpful!**


---

*Made with ❤️ by Ashish Saha*

</div>
