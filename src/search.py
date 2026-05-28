"""
Lógica de busca semântica.

Conecta ao banco vetorial (PGVector), busca os trechos mais relevantes
para a pergunta do usuário e gera uma resposta usando a LLM Gemini,
restrita apenas ao contexto encontrado.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_embeddings")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_vector_store():
    """Cria e retorna a instância do vector store conectada ao PostgreSQL."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def get_llm():
    """Cria e retorna a instância da LLM Gemini."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
    )


def build_prompt(contexto: str, pergunta: str) -> str:
    """Monta o prompt completo com contexto e pergunta."""
    return PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)


def search_and_answer(question: str) -> str:
    """
    Recebe uma pergunta, busca contexto no banco vetorial e retorna
    a resposta da LLM baseada apenas no contexto encontrado.

    Passos:
    1. Vetorizar a pergunta
    2. Buscar os 10 resultados mais relevantes (k=10)
    3. Montar o prompt com o contexto concatenado
    4. Chamar a LLM e retornar a resposta
    """
    # 1. Buscar os 10 resultados mais relevantes
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(question, k=10)

    # 2. Concatenar o contexto dos documentos encontrados
    contexto = "\n\n".join([doc.page_content for doc, score in results])

    # 3. Montar o prompt
    prompt = build_prompt(contexto, question)

    # 4. Chamar a LLM
    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content