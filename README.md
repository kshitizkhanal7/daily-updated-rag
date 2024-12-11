# Daily updated RAG System with PostgreSQL Vector Database

A Retrieval-Augmented Generation (RAG) system that automatically processes text documents from Google Drive, stores embeddings in PostgreSQL, and provides a question-answering interface.

## Features
- Automatic document processing from Google Drive
- Vector similarity search using pgvector
- Real-time question answering through Gradio interface
- Daily automated checks for new documents via GitHub Actions

## Prerequisites
- Supabase account (for PostgreSQL database)
- Google Cloud account with Drive API enabled
- Python 3.10 or higher
