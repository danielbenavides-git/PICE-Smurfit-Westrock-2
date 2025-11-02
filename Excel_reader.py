# =================================================
# DEJAR COMENTADO SI YA SE TIENE EL DATAFRAME FINAL
# =================================================

import pandas as pd

def read_excel_files(df1, df2, df3):
    df_excel = pd.concat([
        pd.read_excel(df1),
        pd.read_excel(df2),
        pd.read_excel(df3)
    ], ignore_index=True)

    columns_to_drop = [
        "Número Documento Referencia", "Material", "Número de Cuenta",
        "Acreedor", "Número Documento", "Descripción", "Documento Compras",
        "Pos Docum Compras", "Activo Fijo", "Clase de Documento",
        "Clase de Actividad", "Deudor", "Elemento PEP", "Orden", "Pedido Cliente",
        "Fecha Valor", "Fecha Entrada", "Fecha Documento", "División",
        "Período", " Año"
    ]

    df_excel = (df_excel
                .dropna(subset=["Número Documento Referencia"])
                .drop(columns=columns_to_drop, errors='ignore'))
    
    return df_excel