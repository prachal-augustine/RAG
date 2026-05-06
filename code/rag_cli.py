import os
import argparse
from read_data_source import (
    search_data_files,
    read_data_files,
    character_text_splitter,
    pdfs_chunk_dict_to_doc,
    embed_index_and_store,
    add_context_to_prompt,
    response_generation,
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
BUFFER_SIZE = 5


def build_index(data_path):
    print(f"Scanning for PDFs in: {data_path}")
    path_of_files = search_data_files(data_path)
    if not path_of_files:
        print("No PDF files found. Exiting.")
        return None
    data_text_dict = read_data_files(path_of_files)
    pdfs_split_text = character_text_splitter(data_text_dict)
    pdfs_documents = pdfs_chunk_dict_to_doc(pdfs_split_text)
    return embed_index_and_store(pdfs_documents)


def format_history(history):
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)


def interactive_query(vector_db):
    print("\nRAG is ready. Type your question or 'exit' to quit.\n")
    history = []
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        history_text = format_history(history) if history else None
        prompt = add_context_to_prompt(query, vector_db, memory=history_text)
        response = response_generation(prompt)
        print(f"\nAssistant: {response}\n")

        history.append((query, response))
        if len(history) > BUFFER_SIZE:
            history.pop(0)


def main():
    parser = argparse.ArgumentParser(description="Interactive RAG CLI")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the folder containing PDFs (default: ../data)",
    )
    args = parser.parse_args()

    vector_db = build_index(args.data_path)
    if vector_db is None:
        return

    interactive_query(vector_db)


if __name__ == "__main__":
    main()
