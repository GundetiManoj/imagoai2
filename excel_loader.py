import pandas as pd

def parse_excel_file(file_path):
    excel_file = pd.ExcelFile(file_path)
    sheet_data = {}

    for sheet_name in excel_file.sheet_names:
        df = excel_file.parse(sheet_name)
        sheet_data[sheet_name] = {
            "dataframe": df,
            "columns": df.columns.tolist(),
            "shape": df.shape
        }

    return sheet_data
