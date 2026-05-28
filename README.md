# Desafio MBA Engenharia de Software com IA - Full Cycle

## Ingestão e Busca Semântica com LangChain e Postgres

Sistema RAG (Retrieval-Augmented Generation) que ingere um PDF, armazena os vetores no PostgreSQL com pgVector e permite buscas semânticas via CLI.

### Tecnologias

- **Linguagem:** Python
- **Framework:** LangChain
- **Banco de dados:** PostgreSQL + pgVector
- **LLM:** Google Gemini (`gemini-2.5-flash-lite`)
- **Embeddings:** Google (`models/embedding-001`)
- **Infra:** Docker & Docker Compose

---

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API do Google (Gemini)

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` e preencha sua chave de API:

```env
GOOGLE_API_KEY=sua-chave-aqui
```

As demais variáveis já possuem valores padrão funcionais.

### 5. Subir o banco de dados

```bash
docker compose up -d
```

Aguarde o banco ficar pronto (healthcheck automático). A extensão `vector` é criada automaticamente.

### 6. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada:

```
📄 Carregando PDF: document.pdf
   → 34 páginas carregadas
   → X chunks gerados (size=1000, overlap=150)
🔗 Conectando ao banco: localhost:5432/rag
🧠 Modelo de embeddings: models/embedding-001
✅ X chunks armazenados com sucesso no PostgreSQL!
```

### 7. Rodar o chat interativo

```bash
python src/chat.py
```

Exemplo de uso:

```
PERGUNTA: Qual o faturamento da empresa?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair`, `exit` ou `quit` para encerrar.

---

## Executar Testes

```bash
pytest tests/ -v
```

---

## Estrutura do Projeto

```
├── docker-compose.yml      # PostgreSQL + pgVector
├── requirements.txt         # Dependências Python
├── .env.example             # Template de variáveis de ambiente
├── spec.md                  # Especificação técnica do projeto
├── document.pdf             # PDF para ingestão (34 páginas)
├── src/
│   ├── ingest.py            # Script de ingestão do PDF
│   ├── search.py            # Lógica de busca semântica + LLM
│   └── chat.py              # CLI para interação com usuário
├── tests/
│   ├── test_ingest.py       # Testes de ingestão
│   ├── test_search.py       # Testes de busca
│   └── test_chat.py         # Testes do CLI
└── README.md                # Este arquivo
```

---

## Parâmetros de Configuração

| Variável | Descrição | Valor Padrão |
|----------|-----------|-------------|
| `GOOGLE_API_KEY` | Chave de API do Google Gemini | *(obrigatório)* |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings | `models/embedding-001` |
| `DATABASE_URL` | Connection string do PostgreSQL | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da coleção no vector store | `pdf_embeddings` |
| `PDF_PATH` | Caminho para o arquivo PDF | `document.pdf` |# mba-ia-desafio-ingestao-busca
# mba-ia-desafio-ingestao-busca
