import os
import gradio as gr
import psycopg2
from sentence_transformers import SentenceTransformer
from transformers import pipeline

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2')

db_params = {
    'host': os.getenv('SUPABASE_HOST'),
    'dbname': 'postgres',
    'user': 'postgres.fhoyagohhletichibcgm',
    'password': os.getenv('SUPABASE_PASSWORD'),
    'port': '6543',
    'sslmode': 'require'
}

def get_context(query, conn):
    query_embedding = embedding_model.encode(query)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT 3
        """, (query_embedding.tolist(),))
        return " ".join([row[0] for row in cur.fetchall()])

def chat(message, history):
    try:
        conn = psycopg2.connect(**db_params)
        context = get_context(message, conn)
        answer = qa_model(question=message, context=context)
        return answer['answer']
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close() if 'conn' in locals() else None

demo = gr.ChatInterface(
    fn=chat,
    title="Document Q&A Bot",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
