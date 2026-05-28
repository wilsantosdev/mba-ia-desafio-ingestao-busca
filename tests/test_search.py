"""
Testes unitários para o módulo de busca (search.py).
"""

import os
import sys
import pytest

# Adicionar src ao path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from search import PROMPT_TEMPLATE, build_prompt


class TestPromptTemplate:
    """Testes para o template de prompt."""

    def test_template_has_contexto_placeholder(self):
        """Verifica que o template contém o placeholder {contexto}."""
        assert "{contexto}" in PROMPT_TEMPLATE

    def test_template_has_pergunta_placeholder(self):
        """Verifica que o template contém o placeholder {pergunta}."""
        assert "{pergunta}" in PROMPT_TEMPLATE

    def test_template_contains_rules(self):
        """Verifica que o prompt contém as regras obrigatórias."""
        assert "Responda somente com base no CONTEXTO" in PROMPT_TEMPLATE
        assert "Nunca invente ou use conhecimento externo" in PROMPT_TEMPLATE
        assert "Nunca produza opiniões ou interpretações" in PROMPT_TEMPLATE

    def test_template_contains_fallback_response(self):
        """Verifica que o template contém a resposta padrão para perguntas fora do contexto."""
        assert "Não tenho informações necessárias para responder sua pergunta" in PROMPT_TEMPLATE

    def test_template_contains_out_of_context_examples(self):
        """Verifica que o template contém os exemplos de perguntas fora do contexto."""
        assert "Qual é a capital da França?" in PROMPT_TEMPLATE
        assert "Quantos clientes temos em 2024?" in PROMPT_TEMPLATE
        assert "Você acha isso bom ou ruim?" in PROMPT_TEMPLATE

    def test_template_contains_instruction(self):
        """Verifica que o template contém a instrução final."""
        assert 'RESPONDA A "PERGUNTA DO USUÁRIO"' in PROMPT_TEMPLATE


class TestBuildPrompt:
    """Testes para a função build_prompt."""

    def test_build_prompt_inserts_contexto(self):
        """Verifica que o contexto é inserido corretamente no prompt."""
        contexto = "O faturamento da empresa foi de 10 milhões."
        pergunta = "Qual o faturamento?"
        prompt = build_prompt(contexto, pergunta)
        assert contexto in prompt

    def test_build_prompt_inserts_pergunta(self):
        """Verifica que a pergunta é inserida corretamente no prompt."""
        contexto = "Texto de exemplo."
        pergunta = "Qual o faturamento da empresa?"
        prompt = build_prompt(contexto, pergunta)
        assert pergunta in prompt

    def test_build_prompt_maintains_rules(self):
        """Verifica que as regras são mantidas após formatação."""
        prompt = build_prompt("contexto teste", "pergunta teste")
        assert "Responda somente com base no CONTEXTO" in prompt
        assert "Nunca invente ou use conhecimento externo" in prompt

    def test_build_prompt_with_empty_contexto(self):
        """Verifica que o prompt funciona com contexto vazio."""
        prompt = build_prompt("", "pergunta teste")
        assert "CONTEXTO:" in prompt
        assert "pergunta teste" in prompt

    def test_build_prompt_with_special_characters(self):
        """Verifica que o prompt lida com caracteres especiais no contexto."""
        contexto = "Valor: R$ 1.000,00 — aumento de 15%"
        pergunta = "Qual o valor?"
        prompt = build_prompt(contexto, pergunta)
        assert contexto in prompt
