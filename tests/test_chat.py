"""
Testes unitários para o módulo de chat (chat.py).
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Adicionar src ao path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chat import main, EXIT_COMMANDS


class TestExitCommands:
    """Testes para os comandos de saída."""

    def test_exit_commands_defined(self):
        """Verifica que os comandos de saída estão definidos."""
        assert "sair" in EXIT_COMMANDS
        assert "exit" in EXIT_COMMANDS
        assert "quit" in EXIT_COMMANDS

    @patch("chat.search_and_answer")
    @patch("builtins.input", side_effect=["sair"])
    def test_sair_exits_loop(self, mock_input, mock_search):
        """Verifica que 'sair' encerra o loop."""
        main()
        mock_search.assert_not_called()

    @patch("chat.search_and_answer")
    @patch("builtins.input", side_effect=["exit"])
    def test_exit_exits_loop(self, mock_input, mock_search):
        """Verifica que 'exit' encerra o loop."""
        main()
        mock_search.assert_not_called()

    @patch("chat.search_and_answer")
    @patch("builtins.input", side_effect=["quit"])
    def test_quit_exits_loop(self, mock_input, mock_search):
        """Verifica que 'quit' encerra o loop."""
        main()
        mock_search.assert_not_called()

    @patch("chat.search_and_answer")
    @patch("builtins.input", side_effect=["SAIR"])
    def test_exit_case_insensitive(self, mock_input, mock_search):
        """Verifica que o comando de saída é case-insensitive."""
        main()
        mock_search.assert_not_called()


class TestInputHandling:
    """Testes para tratamento de input."""

    @patch("chat.search_and_answer")
    @patch("builtins.input", side_effect=["", "", "sair"])
    def test_empty_input_skipped(self, mock_input, mock_search):
        """Verifica que input vazio não gera chamada ao search."""
        main()
        mock_search.assert_not_called()

    @patch("chat.search_and_answer", return_value="Resposta teste")
    @patch("builtins.input", side_effect=["Qual o faturamento?", "sair"])
    def test_valid_question_calls_search(self, mock_input, mock_search):
        """Verifica que uma pergunta válida chama search_and_answer."""
        main()
        mock_search.assert_called_once_with("Qual o faturamento?")

    @patch("chat.search_and_answer", side_effect=Exception("Erro de conexão"))
    @patch("builtins.input", side_effect=["pergunta teste", "sair"])
    def test_error_handling(self, mock_input, mock_search):
        """Verifica que erros são tratados e o loop continua."""
        # Não deve lançar exceção
        main()

    @patch("chat.search_and_answer")
    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_exits(self, mock_input, mock_search):
        """Verifica que Ctrl+C encerra o chat graciosamente."""
        # Não deve lançar exceção
        main()
