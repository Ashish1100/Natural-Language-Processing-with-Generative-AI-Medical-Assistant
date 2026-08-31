# Natural Language Processing with Generative AI: Medical Assistant 🩺

## Overview

Medical Assistant v2 is an intelligent, conversational AI system designed to process, analyze, and retrieve complex healthcare information. Built upon a 3-stage Retrieval-Augmented Generation (RAG) pipeline, this project leverages advanced Natural Language Processing (NLP) and Google's Generative AI to provide accurate, context-aware answers to medical queries based on ingested documents.

## Key Features

- **Advanced RAG Pipeline**: Efficiently retrieves relevant medical data from large document corpora before passing context to the LLM.
- **High-Accuracy Semantic Search**: Utilizes Google's `text-embedding-004` to accurately match user queries with underlying medical literature.
- **Vector Database Integration**: Employs ChromaDB for fast, scalable, and persistent storage of high-dimensional document embeddings.
- **Conversational Interface**: Processes natural language queries to extract medical entities and provide concise, reliable answers.

## Technology Stack

- **Language**: Python  
- **Large Language Model (LLM)**: HF API  
- **Embeddings**: Google `text-embedding-004`  
- **Vector Database**: ChromaDB  
- **Data Processing**: Pandas, NumPy  

## Architecture Workflow

1. **Document Ingestion**: Medical texts and documents are cleaned and split into manageable semantic chunks.  
2. **Embedding Generation**: Chunks are converted into vector representations using embedding models.  
3. **Vector Storage**: Embeddings are indexed and stored in ChromaDB.  
4. **Retrieval & Generation**: User queries are embedded, matched against the vector store using similarity search, and the relevant context is passed to the Gemini LLM to generate a precise response.  


## Future Enhancements

- Integration of specialized medical LLMs (e.g., Med-PaLM) for highly clinical inferences.  
- Adding a user-friendly frontend dashboard (e.g., Streamlit or React).  
- Implementation of conversation memory for multi-turn diagnostic dialogue.  

## 👨‍💻 Author

**Ashish**  

- LinkedIn: [https://linkedin.com/in/ashishsaha21/](https://linkedin.com/in/ashishsaha21/)  
<!-- Content based on your provided project description. [conversation_history:1] -->
