"""
CLI interativo para perguntas sobre o documento PDF.

Utiliza a busca semântica do módulo search.py para responder
perguntas do usuário baseadas apenas no conteúdo do PDF ingerido.
"""

from search import search_and_answer

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main():
    """Loop principal do chat interativo."""
    print("=" * 60)
    print("  🤖 Chat RAG — Pergunte sobre o documento PDF")
    print("  Digite 'sair' ou 'exit' para encerrar")
    print("=" * 60)

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()

            if not question:
                continue

            if question.lower() in EXIT_COMMANDS:
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