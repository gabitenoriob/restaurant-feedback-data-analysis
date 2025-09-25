import os
import pandas as pd
import psycopg2
from google.cloud import bigquery
import pandas_gbq
from datetime import datetime
from dotenv import load_dotenv

from limpeza_dados import limpeza_dados
load_dotenv()


print("Rodando a Cloud Function de ETL de Feedbacks")
# --- Função Principal da Cloud Function ---
def run_etl(request):
    """
    Função principal do ETL, acionada por um gatilho HTTP.
    """
    print("Iniciando o processo de ETL de feedbacks.")
    
    try:
        # --- ETAPA DE EXTRAÇÃO ---
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            database=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASS")
        )
        #Atualiza por dia
        query = "SELECT * FROM public.feedback WHERE timestamp > NOW() - INTERVAL '1 day';"
        df_bruto = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Extração concluída. {len(df_bruto)} registros encontrados.")

        if df_bruto.empty:
            print("Nenhum dado novo para processar. Encerrando.")
            return ("Nenhum dado novo.", 200)

        # --- ETAPA DE TRANSFORMAÇÃO ---
        df_final = limpeza_dados(df_bruto)
        print("Limpeza e transformação dos dados concluídas.")


        # --- ETAPA DE CARGA ---
        project_id = os.environ.get("GCP_PROJECT_ID")
        tabela_destino = "restaurant_feedback.fato_feedbacks" 
        
        pandas_gbq.to_gbq(
            df_final,
            tabela_destino,
            project_id=project_id,
            if_exists='append'
        )
        print(f"Carga concluída. {len(df_final)} registros carregados no BigQuery.")
        
        return ("ETL executado com sucesso!", 200)

    except Exception as e:
        print(f"ERRO no ETL: {e}")
        # É importante retornar um erro 500 para que o Cloud Scheduler saiba que falhou.
        return ("Erro no ETL", 500)
    

run_etl(None)  # Para testes locais, remova ou comente esta linha ao implantar na nuvem.