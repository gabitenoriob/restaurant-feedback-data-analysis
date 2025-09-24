import os
import pandas as pd
import psycopg2
from google.cloud import bigquery
import pandas_gbq
from datetime import datetime



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
        # Pega feedbacks das últimas 24 horas para garantir que não se perca nada.
        query = "SELECT * FROM public.feedbacks WHERE created_at > NOW() - INTERVAL '1 day';"
        df_bruto = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Extração concluída. {len(df_bruto)} registros encontrados.")

        if df_bruto.empty:
            print("Nenhum dado novo para processar. Encerrando.")
            return ("Nenhum dado novo.", 200)

        # --- ETAPA DE TRANSFORMAÇÃO ---
        # df_bruto['sentimento_servico'] = df_bruto['serviceComment'].apply(analisar_sentimento)
        # df_bruto['categoria_nps'] = df_bruto['recommendationRating'].apply(calcula_nps)
        
        # Seleciona e renomeia colunas para o Data Warehouse
        df_final = df_bruto[['id', 'attendantName', 'serviceRating', 'categoria_nps', 'sentimento_servico']]
        df_final = df_final.rename(columns={'id': 'id_feedback_origem'})
        df_final['data_carga_dw'] = datetime.utcnow()
        print("Transformação concluída.")

        # --- ETAPA DE CARGA ---
        project_id = os.environ.get("GCP_PROJECT_ID")
        tabela_destino = "seu_dataset.fato_feedbacks" # <-- MUDE AQUI
        
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