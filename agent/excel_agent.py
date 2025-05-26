from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from agent.tools import build_tools
from agent.rag_retriever import load_sheet_rag
from dotenv import load_dotenv

load_dotenv()

def create_excel_agent(sheet_dfs, sheet_names, rag_retrievers):
    # ✅ build_tools now returns Tool objects directly
    tools = build_tools(sheet_dfs)

    # ✅ Wrap RAG retrievers as Tool objects
    for sheet_name, retriever in rag_retrievers.items():
        tools.append(
            Tool(
                name=f"RAG_{sheet_name}",
                func=retriever,
                description=f"RAG retriever for sheet '{sheet_name}' to answer contextual questions"
            )
        )

    llm = ChatOpenAI(
        temperature=0.3,
        model="gpt-4o"
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent
