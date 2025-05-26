def execute_python_code(code: str):
    try:
        # WARNING: `eval` is dangerous if untrusted input is passed.
        result = eval(code, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Execution error: {e}"
