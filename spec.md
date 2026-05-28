# Spec — Ingestão e Busca Semântica com LangChain e Postgres

## 1. Visão Geral

Sistema RAG (Retrieval-Augmented Generation) que:

1. **Ingestão** — lê um PDF, divide em chunks, gera embeddings e armazena no PostgreSQL + pgVector.
2. **Busca** — recebe perguntas do usuário via CLI, busca os 10 trechos mais relevantes no banco vetorial e gera uma resposta usando uma LLM, restrita apenas ao contexto encontrado.

---

## 2. Decisão de Provider: Gemini (Google)

O projeto suportará **Gemini** como provider principal, conforme recomendado na descrição do desafio:

| Recurso           | Modelo / Pacote                                             |
|--------------------|-------------------------------------------------------------|
| Embeddings         | `models/embedding-001` via `langchain_google_genai.GoogleGenerativeAIEmbeddings` |
| LLM (respostas)    | `gemini-2.5-flash-lite` via `langchain_google_genai.ChatGoogleGenerativeAI`       |
| API Key env var    | `GOOGLE_API_KEY`                                            |

> **Nota:** O `.env.example` já suporta tanto `GOOGLE_API_KEY` quanto `OPENAI_API_KEY`. A implementação será focada no Gemini, mas o código será modular o suficiente para trocar de provider alterando apenas a instanciação dos modelos.

---

## 3. Arquitetura e Fluxo de Dados

```
document.pdf
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ PyPDFLoader │────▶│ RecursiveCharText │────▶│ GoogleGenerativeAI   │
│ (lê o PDF)  │     │ Splitter          │     │ Embeddings           │
└─────────────┘     │ chunk=1000        │     │ (models/embedding-001│
                    │ overlap=150       │     └──────────┬───────────┘
                    └──────────────────┘                 │
                                                         ▼
                                              ┌──────────────────────┐
                                              │ PostgreSQL + pgVector│
                                              │ (PGVector store)     │
                                              └──────────────────────┘
                                                         │
                                                         ▼
                    ┌──────────────────┐     ┌──────────────────────┐
                    │ CLI (chat.py)    │────▶│ similarity_search     │
                    │ input do usuário │     │ _with_score(q, k=10) │
                    └──────────────────┘     └──────────┬───────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │ Prompt Template      │
                                              │ + ChatGoogleGenAI    │
                                              │ (gemini-2.5-flash-   │
                                              │  lite)               │
                                              └──────────────────────┘
                                                         │
                                                         ▼
                                                   Resposta ao
                                                    Usuário
```

---

## 4. Estrutura de Arquivos

```
├── docker-compose.yml          # (já existe) PostgreSQL + pgVector
├── requirements.txt            # (já existe) Dependências pinadas
├── .env.example                # (já existe) Template de variáveis
├── .env                        # (criar) Variáveis reais (git-ignored)
├── spec.md                     # (este arquivo) Especificação
├── document.pdf                # (já existe) PDF para ingestão — 34 páginas
├── src/
│   ├── ingest.py               # (implementar) Script de ingestão
│   ├── search.py               # (implementar) Lógica de busca + prompt
│   ├── chat.py                 # (implementar) CLI interativo
├── tests/
│   ├── test_ingest.py          # Testes para ingestão
│   ├── test_search.py          # Testes para busca
│   └── test_chat.py            # Testes para CLI
├── README.md                   # (atualizar) Instruções de execução
└── doc/                        # (criar se necessário) Documentação
```

---

## 5. Detalhamento da Implementação

### 5.1 `src/ingest.py` — Ingestão do PDF

**Responsabilidades:**
1. Carregar variáveis de ambiente (`.env`)
2. Ler o PDF usando `PyPDFLoader`
3. Dividir em chunks com `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`
4. Criar instância de `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`
5. Conectar ao PostgreSQL e armazenar os vetores via `PGVector`
6. Exibir progresso/resultado no terminal

**Código planejado:**

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_embeddings")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")


def ingest_pdf():
    # 1. Carregar PDF
    print(f"📄 Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"   → {len(documents)} páginas carregadas")

    # 2. Dividir em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   → {len(chunks)} chunks gerados")

    # 3. Criar embeddings e armazenar no PGVector
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    vector_store.add_documents(chunks)
    print(f"✅ {len(chunks)} chunks armazenados com sucesso no PostgreSQL!")


if __name__ == "__main__":
    ingest_pdf()
```

**Pontos de atenção:**
- A connection string deve usar o driver `psycopg` (v3): `postgresql+psycopg://postgres:postgres@localhost:5432/rag`
- `use_jsonb=True` para melhor performance de metadados
- O `PGVector` do `langchain_postgres` cria as tabelas automaticamente

---

### 5.2 `src/search.py` — Lógica de Busca

**Responsabilidades:**
1. Conectar ao banco vetorial (PGVector) com as mesmas configurações de embedding
2. Receber uma pergunta e buscar os `k=10` resultados mais relevantes
3. Montar o prompt usando o template obrigatório
4. Chamar a LLM (Gemini) e retornar a resposta

**Código planejado:**

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_embeddings")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")

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
    """Cria e retorna a instância da LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
    )


def search_and_answer(question: str) -> str:
    """
    Recebe uma pergunta, busca contexto no banco vetorial e retorna
    a resposta da LLM baseada apenas no contexto encontrado.
    """
    # 1. Buscar os 10 resultados mais relevantes
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(question, k=10)

    # 2. Concatenar o contexto
    contexto = "\n\n".join([doc.page_content for doc, score in results])

    # 3. Montar o prompt
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)

    # 4. Chamar a LLM
    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content
```

**Pontos de atenção:**
- O `search_prompt()` original será substituído por `search_and_answer()` que faz o fluxo completo
- O `similarity_search_with_score` retorna tuplas `(Document, score)`, onde menor score = maior similaridade
- O template de prompt é exatamente o especificado no enunciado

---

### 5.3 `src/chat.py` — CLI Interativo

**Responsabilidades:**
1. Inicializar as dependências (verificar conexão)
2. Loop interativo lendo perguntas do usuário
3. Chamar `search_and_answer()` e exibir a resposta
4. Suportar saída com "sair", "exit" ou Ctrl+C

**Código planejado:**

```python
from search import search_and_answer


def main():
    print("=" * 60)
    print("  🤖 Chat RAG — Pergunte sobre o documento PDF")
    print("  Digite 'sair' ou 'exit' para encerrar")
    print("=" * 60)

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()

            if not question:
                continue

            if question.lower() in ("sair", "exit", "quit"):
                print("\n👋 Até mais!")
                break

            response = search_and_answer(question)
            print(f"\nRESPOSTA: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Até mais!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")


if __name__ == "__main__":
    main()
```

---

## 6. Configuração do Ambiente (`.env`)

Valores a serem preenchidos no `.env` (baseado no `.env.example`):

```env
GOOGLE_API_KEY=<sua-chave-api-google>
GOOGLE_EMBEDDING_MODEL=models/embedding-001
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=pdf_embeddings
PDF_PATH=document.pdf
```

> **IMPORTANTE:** A connection string DEVE usar `postgresql+psycopg://` (driver psycopg3), que é o que o `langchain_postgres.PGVector` espera.

---

## 7. Plano de Testes

### 7.1 `tests/test_ingest.py`

| Teste                          | Descrição                                                                |
|--------------------------------|--------------------------------------------------------------------------|
| `test_pdf_loading`             | Verifica que o PDF é carregado e retorna documentos com conteúdo         |
| `test_text_splitting`          | Verifica que os chunks respeitam `chunk_size=1000` e `overlap=150`       |
| `test_chunks_have_metadata`    | Verifica que os chunks mantêm metadados (página, source)                 |
| `test_chunk_count_reasonable`  | Verifica que o número de chunks é razoável para um PDF de 34 páginas     |

### 7.2 `tests/test_search.py`

| Teste                              | Descrição                                                            |
|-------------------------------------|----------------------------------------------------------------------|
| `test_prompt_template_format`       | Verifica que o template aceita `{contexto}` e `{pergunta}`           |
| `test_prompt_contains_rules`        | Verifica que o prompt contém as regras obrigatórias                   |
| `test_prompt_contains_examples`     | Verifica que o prompt contém os exemplos de perguntas fora do contexto|

### 7.3 `tests/test_chat.py`

| Teste                          | Descrição                                                                |
|--------------------------------|--------------------------------------------------------------------------|
| `test_exit_commands`           | Verifica que "sair", "exit", "quit" encerram o loop                      |
| `test_empty_input_skipped`     | Verifica que input vazio não gera chamada ao search                      |

> **Nota:** Testes de integração (que exigem banco de dados e API key) serão marcados como `@pytest.mark.integration` e rodarão separadamente.

---

## 8. Dependências

Todas as dependências já estão no `requirements.txt` existente. As principais são:

| Pacote                        | Versão  | Uso                          |
|-------------------------------|---------|------------------------------|
| `langchain`                   | 0.3.27  | Framework principal          |
| `langchain-community`         | 0.3.27  | PyPDFLoader                  |
| `langchain-text-splitters`    | 0.3.9   | RecursiveCharacterTextSplitter|
| `langchain-google-genai`      | 2.1.9   | Embeddings + LLM Gemini      |
| `langchain-openai`            | 0.3.30  | (alternativa) OpenAI         |
| `langchain-postgres`          | 0.0.15  | PGVector store               |
| `psycopg`/`psycopg-binary`   | 3.2.9   | Driver PostgreSQL v3         |
| `pgvector`                    | 0.3.6   | Extensão pgvector Python     |
| `pypdf`                       | 6.0.0   | Leitura de PDFs              |
| `python-dotenv`               | 1.1.1   | Carregamento de .env         |

---

## 9. Infraestrutura (Docker)

O `docker-compose.yml` já está configurado com:

- **PostgreSQL 17** com imagem `pgvector/pgvector:pg17`
- **Extensão `vector`** criada automaticamente pelo serviço `bootstrap_vector_ext`
- **Porta:** `5432`
- **Credenciais:** `postgres` / `postgres` / banco `rag`
- **Volume persistente:** `postgres_data`

Basta executar `docker compose up -d` para ter o banco pronto.

---

## 10. Ordem de Execução

```bash
# 1. Criar e ativar virtualenv
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar .env (copiar de .env.example e preencher)
cp .env.example .env
# editar .env com suas chaves de API

# 4. Subir o banco de dados
docker compose up -d

# 5. Executar ingestão do PDF
python src/ingest.py

# 6. Rodar o chat interativo
python src/chat.py

# 7. Executar testes
pytest tests/ -v
```

---

## 11. Checklist de Entrega

- [ ] `src/ingest.py` — Ingestão completa do PDF com chunks e embeddings
- [ ] `src/search.py` — Busca vetorial + montagem de prompt + chamada LLM
- [ ] `src/chat.py` — CLI interativo funcional
- [ ] `.env.example` — Template completo das variáveis
- [ ] `tests/` — Testes unitários para cada módulo
- [ ] `README.md` — Instruções claras de execução
- [ ] Validação end-to-end: pergunta → resposta baseada no PDF
- [ ] Validação de pergunta fora do contexto → resposta padrão

---

## 12. Decisões Confirmadas

| # | Decisão | Resposta |
|---|---------|----------|
| 1 | **Provider** | ✅ Apenas **Gemini** (Google) — sem suporte a OpenAI |
| 2 | **Modelo LLM** | ✅ `gemini-2.5-flash-lite` — o mais barato que atende o caso de uso |
| 3 | **Testes** | ✅ Unitários (sem dependências externas) + integração com marker `@pytest.mark.integration` |
| 4 | **PDF** | ✅ `document.pdf` já existente no repositório (34 páginas, ~175KB) |
