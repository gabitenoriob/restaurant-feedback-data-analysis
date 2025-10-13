# =====================================================
# VISUALIZAÇÃO DE INSIGHTS E GRÁFICOS AUTOMÁTICOS
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def plotagens(df_final):
    # Configurações estéticas
    sns.set(style="whitegrid", palette="viridis", font_scale=1.1)

    # =====================================
    # 1️⃣ DISTRIBUIÇÃO DE SENTIMENTOS POR ASPECTO
    # =====================================
    plt.figure(figsize=(10,5))
    sent_cols = ['sentimento_comida', 'sentimento_servico', 'sentimento_ambiente']
    df_melt = df_final.melt(value_vars=sent_cols, var_name='aspecto', value_name='sentimento')
    sns.countplot(data=df_melt, x='aspecto', hue='sentimento')
    plt.title("Distribuição de Sentimentos por Aspecto")
    plt.xlabel("Aspecto")
    plt.ylabel("Quantidade de Comentários")
    plt.show()

    # =====================================
    # 2️⃣ COMPARAÇÃO NPS REAL VS PREVISTO
    # =====================================
    plt.figure(figsize=(8,5))
    sns.histplot(df_final[['recommendation_rating','nps_pred']], kde=True, bins=20)
    plt.title("Distribuição NPS Real vs Previsto")
    plt.xlabel("Nota de Recomendação")
    plt.show()

    # =====================================
    # 3️⃣ DIFERENÇA ENTRE NPS REAL E PREVISTO
    # =====================================
    plt.figure(figsize=(8,5))
    sns.histplot(df_final['dif_nps'], kde=True, color='darkblue')
    plt.title("Diferença entre NPS Real e Previsto (dif_nps)")
    plt.xlabel("Diferença (Positivo = previsão maior que real)")
    plt.show()

    # =====================================
    # 4️⃣ DISTRIBUIÇÃO DE CHURN
    # =====================================
    plt.figure(figsize=(6,4))
    sns.countplot(x='churn_pred', data=df_final)
    plt.title("Distribuição de Clientes Churn vs Não-Churn")
    plt.xlabel("Churn (1 = sim, 0 = não)")
    plt.ylabel("Quantidade")
    plt.show()

    # =====================================
    # 5️⃣ CHURN PROBABILIDADE VS DIF_NPS
    # =====================================
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df_final, x='dif_nps', y='churn_prob', hue='categoria_nps')
    plt.title("Correlação entre Diferença NPS e Probabilidade de Churn")
    plt.xlabel("Diferença NPS (Previsto - Real)")
    plt.ylabel("Probabilidade de Churn")
    plt.show()

    # =====================================
    # 6️⃣ SENTIMENTO X NPS
    # =====================================
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df_melt.join(df_final[['recommendation_rating']]), 
                x='aspecto', y='recommendation_rating', hue='sentimento')
    plt.title("Distribuição da Nota de Recomendação por Sentimento e Aspecto")
    plt.show()

    # =====================================
    # 7️⃣ NPS E CHURN POR TÓPICO
    # =====================================
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_final, x='tema', y='recommendation_rating', ci=None)
    plt.title("Média de NPS Real por Tópico")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.barplot(data=df_final, x='tema', y='churn_prob', ci=None)
    plt.title("Probabilidade Média de Churn por Tópico")
    plt.show()

    # =====================================
    # 8️⃣ NPS E CHURN POR DIA DA SEMANA
    # =====================================
    if 'data' in df_final.columns:
        df_final['dia_semana'] = pd.to_datetime(df_final['data']).dt.day_name()

        plt.figure(figsize=(10,5))
        sns.barplot(data=df_final, x='dia_semana', y='recommendation_rating', ci=None)
        plt.title("Média de NPS por Dia da Semana")
        plt.show()

        plt.figure(figsize=(10,5))
        sns.barplot(data=df_final, x='dia_semana', y='churn_prob', ci=None)
        plt.title("Probabilidade Média de Churn por Dia da Semana")
        plt.show()

    # =====================================
    # 9️⃣ COMPARAÇÃO POR ATENDENTE
    # =====================================
    if 'attendente' in df_final.columns:
        plt.figure(figsize=(10,6))
        sns.boxplot(data=df_final, x='attendente', y='recommendation_rating')
        plt.title("Distribuição de NPS por Atendente")
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(10,6))
        sns.barplot(data=df_final, x='attendente', y='churn_prob', ci=None)
        plt.title("Probabilidade Média de Churn por Atendente")
        plt.xticks(rotation=45)
        plt.show()

    # =====================================
    # 🔟 CORRELAÇÕES ENTRE VARIÁVEIS NUMÉRICAS
    # =====================================
    num_cols = ['service_rating', 'food_rating', 'environment_rating',
                'recommendation_rating', 'nps_pred', 'dif_nps', 'churn_prob']
    plt.figure(figsize=(10,6))
    sns.heatmap(df_final[num_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlação entre Notas, NPS e Churn")
    plt.show()
