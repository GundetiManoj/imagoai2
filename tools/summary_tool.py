def summarize_sheet(df, sheet_name="Sheet"):
    num_rows, num_cols = df.shape
    summary = f"Sheet '{sheet_name}' has {num_rows} rows and {num_cols} columns.\n"

    summary += "Columns and sample values:\n"
    for col in df.columns:
        sample_values = df[col].dropna().unique()[:3]
        sample_str = ", ".join(map(str, sample_values))
        summary += f"- {col}: {sample_str}\n"

    return summary
