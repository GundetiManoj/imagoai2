from excel_loader import parse_excel_file
from embeddings_generator import tabular_to_sentences, get_tabular_embeddings, save_outputs
from query_engine import ask_question
from tools.summary_tool import summarize_sheet
from tools.python_tool import execute_python_code

def main():
    file_path = "sample_data/example.xlsx"
    sheet_data = parse_excel_file(file_path)
    sheet_name, df = list(sheet_data.items())[0]
    df = df["dataframe"]

    # Embedding generation
    print(f"\n📊 Processing sheet: {sheet_name}")
    sentences = tabular_to_sentences(df)
    embeddings = get_tabular_embeddings(sentences)
    save_outputs(sentences, embeddings)

    # Show summary at startup
    summary = summarize_sheet(df, sheet_name)
    print("\n🔎 Sheet Summary:\n" + summary)

    while True:
        query = input("\nAsk a question about the Excel file (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        # Tool: If explicitly asking for summary
        if "summary" in query.lower():
            print("\n📝 Sheet Summary:\n", summary)
            continue

        # Tool: If asking to compute Python expression
        if query.lower().startswith("compute:"):
            code = query.replace("compute:", "").strip()
            result = execute_python_code(code)
            print("\n🧮 Computation Result:", result)
            continue

        # Default: Ask via LLM RAG
        answer = ask_question(query)
        print("\n🧠 Answer:\n", answer)

if __name__ == "__main__":
    main()
