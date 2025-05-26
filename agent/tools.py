from langchain.tools import Tool
from tools.summary_tool import summarize_sheet
from tools.python_tool import execute_python_code
def build_tools(sheet_dfs):
    tools = []

    for sheet_name, df in sheet_dfs.items():
        tools.append(
            Tool.from_function(
                name=f"Summary_{sheet_name}",
                func=lambda _: summarize_sheet(df, sheet_name),
                description=f"Provides a summary of sheet '{sheet_name}'"
            )
        )

    tools.append(
        Tool.from_function(
            name="PythonExecutor",
            func=lambda code: execute_python_code(code),
            description="Use this tool to run Python code for computing or analyzing full-sheet data."
        )
    )

    return tools
