"""
Script de ingestão do PDF.

Lê o arquivo PDF, divide em chunks, gera embeddings via Google Gemini
e armazena os vetores no PostgreSQL com pgVector.
"""

import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_embeddings")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")


def load_pdf(pdf_path: str):
    """Carrega o PDF e retorna a lista de documentos (um por página)."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Divide os documentos em chunks usando RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, embeddings, collection_name: str, connection: str):
    """Cria o vector store no PostgreSQL e adiciona os chunks em batches com retry e backoff exponencial."""
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    
    batch_size = 5
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"   → Enviando batch {i//batch_size + 1}/{(total_chunks - 1)//batch_size + 1} ({len(batch)} chunks)...")
        
        retries = 5
        delay = 2
        for attempt in range(retries):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                # Se for erro de quota/rate limit, aguarda com backoff exponencial
                if ("RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)) and attempt < retries - 1:
                    print(f"      ⚠️ Limite de quota atingido. Aguardando {delay}s antes de tentar novamente (Tentativa {attempt + 1}/{retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
                    
        if i + batch_size < total_chunks:
            time.sleep(1)
            
    return vector_store


def ingest_pdf():
    """Pipeline completo de ingestão: carrega PDF, divide em chunks e armazena."""
    if not DATABASE_URL:
        print("❌ Erro: DATABASE_URL não configurada no .env")
        return

    if not os.path.exists(PDF_PATH):
        print(f"❌ Erro: Arquivo PDF não encontrado: {PDF_PATH}")
        return

    # 1. Carregar PDF
    print(f"📄 Carregando PDF: {PDF_PATH}")
    documents = load_pdf(PDF_PATH)
    print(f"   → {len(documents)} páginas carregadas")

    # 2. Dividir em chunks
    chunks = split_documents(documents)
    print(f"   → {len(chunks)} chunks gerados (size=1000, overlap=150)")

    # 3. Criar embeddings e armazenar no PGVector
    print(f"🔗 Conectando ao banco: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else '...'}")
    print(f"🧠 Modelo de embeddings: {EMBEDDING_MODEL}")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    create_vector_store(chunks, embeddings, COLLECTION_NAME, DATABASE_URL)

    print(f"✅ {len(chunks)} chunks armazenados com sucesso no PostgreSQL!")
    print(f"   Coleção: {COLLECTION_NAME}")


if __name__ == "__main__":
    ingest_pdf()