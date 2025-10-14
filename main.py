import os
import pandas as pd
import psycopg2
from google.cloud import bigquery
import pandas_gbq
from datetime import datetime, timezone
from dotenv import load_dotenv
from transformar_dados import transformar_dados
from medicao_churn import medicao_churn
from calcular_nps import calcular_nps
import numpy as np
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
        conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
        #Atualiza por dia
        query = "SELECT * FROM public.feedback WHERE timestamp > NOW() - INTERVAL '10 day';"
        query_attendant = "SELECT id, name FROM public.attendants;"
        print("Executando a query:", query) 
        df_bruto = pd.read_sql_query(query, conn)
        df_atendentes = pd.read_sql_query(query_attendant, conn)
        print(f"colunas: {df_bruto.columns.tolist()}")
        print(f"colunas: {df_atendentes.columns.tolist()}")

        conn.close()
        print(f"Extração concluída. {len(df_bruto)} registros encontrados.")

        if df_bruto.empty:
            print("Nenhum dado novo para processar. Encerrando.")
            return ("Nenhum dado novo.", 200)

        # --- ETAPA DE TRANSFORMAÇÃO ---
        df_bruto = df_bruto.rename(columns={'id': 'id_feedback_origem'})
        df_bruto['data_carga_dw'] = datetime.now(timezone.utc)
        df_bruto['data_feedback'] = df_bruto['timestamp']
        df_bruto['dia_semana'] = df_bruto['timestamp'].dt.day_name()
        df_bruto = df_bruto.drop(columns=['timestamp'])

        df_bruto['attendant_name'] = df_bruto['attendant_id'].map(df_atendentes.set_index('id')['name'])
        df_final, topic_model = transformar_dados(df_bruto)
        document_info = topic_model.get_document_info(df_final['general_comment'])
        document_info = document_info.rename(columns={
            'Topic': 'id_topico',
            'Name': 'nome_topico',
            'Top_n_words': 'palavras_chave_topico',
            'Probability': 'probabilidade_topico',
            'Representative_document': 'documento_representativo'
        })

        df_final_com_topicos = df_final.join(document_info)
        print("Limpeza e transformação dos dados concluídas.")

        #Churn e NPS
        df_final_com_topicos['churn_pred'] = medicao_churn(df_final_com_topicos)
        df_final_com_topicos['nps_pred'] = calcular_nps(df_final_com_topicos)



        # --- ETAPA DE CARGA ---
        project_id = os.environ.get("GCP_PROJECT_ID")
        tabela_destino = "restaurant_feedback.fato_feedbacks" 
        
        pandas_gbq.to_gbq(
            df_final_com_topicos,
            tabela_destino,
            project_id=project_id,
            if_exists='append'
        )
        print(f"Carga concluída. {len(df_final_com_topicos)} registros carregados no BigQuery.")
        
        return ("ETL executado com sucesso!", 200)

    except Exception as e:
        print(f"ERRO no ETL: {e} na etapa {e.__traceback__.tb_lineno}")
        return ("Erro no ETL", 500)
    

#run_etl(None)  # Para testes locais, remova ou comente esta linha ao implantar na nuvem.

if __name__ == "__main__":
    # Gerar dados de exemplo para testar os scripts

    data = pd.DataFrame({
        'id_feedback_origem': range(1, 21),
        'attendant_id': np.random.choice([1, 2, 3, 4], 20),
        'general_comment': [
            "Ótimo atendimento!", "Demorou muito.", "Comida excelente.",
            "Não gostei do ambiente.", "Atendente simpático.",
            "Voltarei mais vezes.", "Preço justo.", "Poucas opções no cardápio.",
            "Recomendo!", "Música alta demais.", "Pedido veio errado.",
            "Ambiente agradável.", "Faltou sobremesa.", "Garçom atencioso.",
            "Demorou para trazer a conta.", "Comida fria.", "Tudo perfeito.",
            "Banheiro sujo.", "Mesa confortável.", "Cardápio variado."
        ],
        'timestamp': pd.date_range(end=datetime.now(), periods=20, freq='D'),
        'nota': np.random.randint(1, 11, 20),
        'recommendation_rating': np.random.randint(1, 11, 20)  # Adiciona a coluna necessária
    })

    # Simular df_atendentes
    df_atendentes = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Ana', 'Bruno', 'Carla', 'Diego']
    })

    # Adicionar colunas esperadas pelo pipeline
    data['data_carga_dw'] = datetime.now(timezone.utc)
    data['data_feedback'] = data['timestamp']
    data['dia_semana'] = data['timestamp'].dt.day_name()
    data['attendant_name'] = data['attendant_id'].map(df_atendentes.set_index('id')['name'])

    # Chamar funções de transformação e predição
    df_final = transformar_dados(data)
    # document_info = topic_model.get_document_info(df_final['general_comment'])
    # document_info = document_info.rename(columns={
    #     'Topic': 'id_topico',
    #     'Name': 'nome_topico',
    #     'Top_n_words': 'palavras_chave_topico',
    #     'Probability': 'probabilidade_topico',
    #     'Representative_document': 'documento_representativo'
    # })

    # df_final_com_topicos = df_final.join(document_info)
    df_final['nps_pred'] = calcular_nps(df_final)
    df_final_com_topicos = medicao_churn(df_final)

    print(df_final_com_topicos.head())
    #salvar como csv
    df_final_com_topicos.to_csv("dados_transformados.csv", index=False)
    print(df_final_com_topicos.columns)


    # Se quiser rodar o ETL completo, descomente:
    # run_etl(None)