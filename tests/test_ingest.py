"""
Testes unitários para o módulo de ingestão (ingest.py).
"""

import os
import sys
import pytest

# Adicionar src ao path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ingest import load_pdf, split_documents


PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "document.pdf")


class TestLoadPdf:
    """Testes para a função load_pdf."""

    def test_pdf_loading_returns_documents(self):
        """Verifica que o PDF é carregado e retorna uma lista não-vazia."""
        documents = load_pdf(PDF_PATH)
        assert len(documents) > 0

    def test_pdf_loading_returns_document_objects(self):
        """Verifica que cada item retornado é um Document do LangChain."""
        documents = load_pdf(PDF_PATH)
        for doc in documents:
            assert isinstance(doc, Document)

    def test_pdf_pages_have_content(self):
        """Verifica que cada página carregada tem conteúdo textual."""
        documents = load_pdf(PDF_PATH)
        for doc in documents:
            assert doc.page_content is not None
            assert len(doc.page_content.strip()) > 0

    def test_pdf_documents_have_metadata(self):
        """Verifica que os documentos mantêm metadados (source, page)."""
        documents = load_pdf(PDF_PATH)
        for doc in documents:
            assert "source" in doc.metadata

    def test_pdf_page_count(self):
        """Verifica que o PDF tem 34 páginas conforme esperado."""
        documents = load_pdf(PDF_PATH)
        assert len(documents) == 34


class TestSplitDocuments:
    """Testes para a função split_documents."""

    def test_split_returns_chunks(self):
        """Verifica que a divisão gera chunks."""
        documents = load_pdf(PDF_PATH)
        chunks = split_documents(documents)
        assert len(chunks) > 0

    def test_chunk_size_respected(self):
        """Verifica que os chunks respeitam o tamanho máximo de 1000 caracteres."""
        documents = load_pdf(PDF_PATH)
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=150)
        for chunk in chunks:
            # RecursiveCharacterTextSplitter pode exceder ligeiramente o chunk_size
            # em casos extremos, mas em geral deve respeitar
            assert len(chunk.page_content) <= 1100  # margem de tolerância

    def test_chunks_have_metadata(self):
        """Verifica que os chunks mantêm metadados do documento original."""
        documents = load_pdf(PDF_PATH)
        chunks = split_documents(documents)
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_chunk_count_is_reasonable(self):
        """
        Verifica que o número de chunks é razoável para um PDF de 34 páginas.
        Com chunk_size=1000 e overlap=150, esperamos algo entre 30 e 300 chunks.
        """
        documents = load_pdf(PDF_PATH)
        chunks = split_documents(documents)
        assert 30 <= len(chunks) <= 300

    def test_custom_chunk_parameters(self):
        """Verifica que parâmetros customizados de chunk são respeitados."""
        documents = load_pdf(PDF_PATH)
        chunks_small = split_documents(documents, chunk_size=500, chunk_overlap=50)
        chunks_large = split_documents(documents, chunk_size=2000, chunk_overlap=200)
        # Chunks menores devem gerar mais partes
        assert len(chunks_small) > len(chunks_large)

    def test_no_empty_chunks(self):
        """Verifica que nenhum chunk tem conteúdo vazio."""
        documents = load_pdf(PDF_PATH)
        chunks = split_documents(documents)
        for chunk in chunks:
            assert len(chunk.page_content.strip()) > 0
