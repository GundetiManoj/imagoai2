from excel_loader import parse_excel_file
from embeddings_generator import tabular_to_sentences, get_tabular_embeddings, save_outputs
from agent.excel_agent import create_excel_agent
from agent.rag_retriever import load_sheet_rag
import os

def prepare_embeddings(sheet_dfs):
    model_name = "all-MiniLM-L6-v2"
    for sheet_name, df in sheet_dfs.items():
        out_dir = os.path.join("outputs", sheet_name)
        sentences = tabular_to_sentences(df)
        embeddings = get_tabular_embeddings(sentences, model_name)
        save_outputs(sentences, embeddings, out_dir)

def main():
    file_path = "sample_data/example.xlsx"
    parsed = parse_excel_file(file_path)
    sheet_dfs = {name: info["dataframe"] for name, info in parsed.items()}

    print("\n📦 Preparing sheet embeddings...")
    prepare_embeddings(sheet_dfs)

    # Load rag retrievers
    rag_retrievers = {
        name: load_sheet_rag(name)
        for name in sheet_dfs.keys()
    }

    # Build the agent
    agent = create_excel_agent(sheet_dfs, list(sheet_dfs.keys()), rag_retrievers)

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        response = agent.run(query)
        print("\n🤖 Response:\n", response)

if __name__ == "__main__":
    main()
