import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings

def plotagens(df):
# --- 2. ANÁLISE GERAL DA SATISFAÇÃO ---
    def analise_geral_satisfacao(df):
        """
        OBJETIVO: Ter uma visão macro da satisfação dos clientes.
        EXPLICAÇÃO: Esta análise mostra a proporção de clientes que promovem, são neutros ou
        detraem a marca (NPS), e também a distribuição geral dos sentimentos extraídos
        dos comentários. É o principal termômetro da saúde da reputação do restaurante.
        AÇÃO GERENCIAL: Se o número de detratores for alto, é um alerta vermelho que exige
        uma investigação mais profunda nos outros gráficos para entender a causa raiz.
        """
        print("Executando Análise Geral da Satisfação...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Gráfico de Categoria NPS
        nps_counts = df['categoria_nps'].value_counts()
        axes[0].pie(nps_counts, labels=nps_counts.index, autopct='%1.1f%%', startangle=90,
                colors=['#4CAF50', '#FFC107', '#F44336'])
        axes[0].set_title('Distribuição de Clientes por Categoria NPS', fontsize=16)
        
        # Gráfico de Sentimento Geral
        sentimento_counts = df['sentimento_geral'].value_counts()
        sns.barplot(x=sentimento_counts.index, y=sentimento_counts.values, ax=axes[1], palette="viridis")
        axes[1].set_title('Distribuição do Sentimento Geral dos Comentários', fontsize=16)
        axes[1].set_ylabel('Quantidade de Feedbacks')
        
        plt.tight_layout()
        plt.show()


    # --- 3. DESEMPENHO POR ASPECTO ---
    def analise_desempenho_aspectos(df):
        """
        OBJETIVO: Identificar os pontos fortes e fracos do restaurante.
        EXPLICAÇÃO: Este gráfico compara o número de menções positivas e negativas para cada
        aspecto principal (Comida, Serviço, Ambiente, Preço). Ele mostra claramente
        o que os clientes mais amam e o que mais odeiam.
        AÇÃO GERENCIAL: Focar os esforços de melhoria no aspecto com mais sentimentos
        negativos. Se o serviço é o problema, invista em treinamento. Se for a comida,
        revise o cardápio ou a cozinha. Elogie a equipe pelo aspecto com mais pontos positivos.
        """
        print("\nExecutando Análise de Desempenho por Aspecto...")
        aspectos = ['sentimento_comida', 'sentimento_servico', 'sentimento_ambiente', 'sentimento_preco']
        sentimentos = ['Positivo', 'Negativo']
        
        # Coletando os dados
        data = []
        for aspecto in aspectos:
            counts = df[aspecto].value_counts()
            for sentimento in sentimentos:
                data.append([aspecto.replace('sentimento_', '').capitalize(), sentimento, counts.get(sentimento, 0)])
                
        df_aspectos = pd.DataFrame(data, columns=['Aspecto', 'Sentimento', 'Contagem'])
        
        # Plotando o gráfico
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_aspectos, x='Aspecto', y='Contagem', hue='Sentimento', palette={'Positivo': 'green', 'Negativo': 'red'})
        plt.title('Contagem de Sentimentos Positivos vs. Negativos por Aspecto', fontsize=16)
        plt.ylabel('Quantidade de Menções')
        plt.xlabel('Aspecto do Restaurante')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


    # --- 4. CORRELAÇÃO E IMPACTO NA NOTA FINAL ---
    def analise_correlacao_impacto(df):
        """
        OBJETIVO: Entender o que mais influencia a nota de recomendação que o cliente dá.
        EXPLICAÇÃO: O heatmap de correlação mostra a relação numérica entre as notas que os
        clientes dão para cada aspecto e a nota final. Quanto mais próximo de 1 (azul escuro),
        mais forte é a relação. Um valor alto entre 'service_rating' e 'recommendation_rating',
        por exemplo, significa que um bom serviço quase sempre leva a uma boa recomendação.
        AÇÃO GERENCIAL: Priorize os investimentos no aspecto com maior correlação. Se o serviço
        tem a correlação mais forte, melhorá-lo é a maneira mais eficaz de aumentar o NPS.
        """
        print("\nExecutando Análise de Correlação e Impacto...")
        cols_rating = ['service_rating', 'food_rating', 'environment_rating', 'recommendation_rating']
        correlation_matrix = df[cols_rating].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=.5)
        plt.title('Correlação entre as Notas dos Aspectos e a Recomendação Final', fontsize=16)
        plt.show()


    # --- 5. ANÁLISE TEMPORAL E DIAS DA SEMANA ---
    def analise_temporal(df):
        """
        OBJETIVO: Identificar se a qualidade está variando ao longo do tempo ou em dias específicos.
        EXPLICAÇÃO: O primeiro gráfico mostra a tendência da nota média de recomendação. A linha
        está subindo ou descendo? O segundo gráfico mostra se há dias da semana com notas
        sistematicamente piores, o que pode indicar problemas de equipe ou superlotação.
        AÇÃO GERENCIAL: Se as notas caem nos fins de semana, pode ser necessário reforçar a equipe.
        Se a tendência geral é de queda, é preciso uma intervenção estratégica urgente.
        """
        print("\nExecutando Análise Temporal...")
        df_temp = df.copy()
        df_temp.set_index('data_feedback', inplace=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 14))
        
        # Gráfico de Tendência da Nota Média
        media_movel = df_temp['recommendation_rating'].rolling(window=7).mean()
        axes[0].plot(df_temp.index, df_temp['recommendation_rating'], marker='o', linestyle='-', alpha=0.3, label='Nota Diária')
        axes[0].plot(media_movel.index, media_movel, color='red', linewidth=2, label='Média Móvel (7 dias)')
        axes[0].set_title('Tendência da Nota de Recomendação ao Longo do Tempo', fontsize=16)
        axes[0].set_ylabel('Nota de Recomendação (0-10)')
        axes[0].legend()
        
        # Gráfico de Notas por Dia da Semana
        dias_ordem = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sns.boxplot(data=df, x='dia_semana', y='recommendation_rating', ax=axes[1], order=dias_ordem, palette="coolwarm")
        axes[1].set_title('Distribuição das Notas de Recomendação por Dia da Semana', fontsize=16)
        axes[1].set_xlabel('Dia da Semana')
        axes[1].set_ylabel('Nota de Recomendação (0-10)')
        
        plt.tight_layout()
        plt.show()


    # --- 6. ANÁLISE DE ATENDENTES ---
    def analise_atendentes(df):
        """
        OBJETIVO: Avaliar o desempenho individual dos atendentes.
        EXPLICAÇÃO: O gráfico mostra a nota média de serviço e de recomendação para cada
        atendente. Isso ajuda a identificar os funcionários de alta performance e aqueles
        que podem precisar de mais treinamento ou feedback.
        AÇÃO GERENCIAL: Reconhecer e recompensar os melhores atendentes. Oferecer treinamento
        e suporte para aqueles com as notas mais baixas.
        """
        print("\nExecutando Análise de Atendentes...")
        df_atendentes = df.groupby('attendant_name').agg(
            media_servico=('service_rating', 'mean'),
            media_recomendacao=('recommendation_rating', 'mean'),
            contagem=('id_feedback_origem', 'count')
        ).reset_index().sort_values('media_servico', ascending=False)
        
        plt.figure(figsize=(15, 8))
        sns.barplot(data=df_atendentes, x='attendant_name', y='media_servico', color='skyblue')
        plt.title('Nota Média de Serviço por Atendente', fontsize=16)
        plt.xlabel('Nome do Atendente')
        plt.ylabel('Nota Média de Serviço (0-10)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    # --- 7. ANÁLISE DE CHURN ---
    def analise_churn(df):
        """
        OBJETIVO: Entender o que leva os clientes a não quererem voltar (churn).
        EXPLICAÇÃO: O gráfico compara o sentimento geral dos clientes que o modelo previu que
        darão churn com aqueles que não darão. Mostra se os clientes com risco de churn
        são predominantemente detratores ou têm sentimentos negativos.
        AÇÃO GERENCIAL: Analisar os comentários dos clientes com alta probabilidade de churn
        para entender as queixas específicas e agir para evitar que futuros clientes
        tenham a mesma experiência negativa.
        """
        print("\nExecutando Análise de Churn...")
        df_churn = df.groupby('churn_pred')['sentimento_geral'].value_counts(normalize=True).mul(100).rename('percentual').reset_index()
        df_churn['churn_pred'] = df_churn['churn_pred'].map({0: 'Voltará', 1: 'Não Voltará'})
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_churn, x='churn_pred', y='percentual', hue='sentimento_geral', palette='pastel')
        plt.title('Composição do Sentimento Geral para Clientes com e sem Risco de Churn', fontsize=16)
        plt.xlabel('Previsão de Churn')
        plt.ylabel('Percentual de Feedbacks (%)')
        plt.tight_layout()
        plt.show()

    # --- 8. NUVEM DE PALAVRAS DAS JUSTIFICATIVAS ---
    def nuvem_palavras(df):
        """
        OBJETIVO: Visualizar rapidamente os termos mais comuns nos elogios e reclamações.
        EXPLICAÇÃO: As nuvens de palavras mostram os termos mais frequentes nas justificativas.
        Palavras maiores aparecem mais vezes. Isso dá um resumo visual e intuitivo do que
        as pessoas estão falando, tanto de bom quanto de ruim.
        AÇÃO GERENCIAL: Usar as palavras-chave da nuvem negativa para guiar reuniões de melhoria.
        Se "lento" e "demora" são grandes, o foco é agilizar o serviço.
        """
        print("\nGerando Nuvens de Palavras...")
        
        # Juntar todas as justificativas
        justificativas = ['justificativa_comida', 'justificativa_servico', 'justificativa_ambiente', 'justificativa_preco']
        sentimentos = ['sentimento_comida', 'sentimento_servico', 'sentimento_ambiente', 'sentimento_preco']
        
        textos_positivos = []
        textos_negativos = []
        
        for i in range(len(justificativas)):
            # Filtra para evitar listas vazias e NaN
            df_filtered = df[df[justificativas[i]].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
            
            positivos = df_filtered[df_filtered[sentimentos[i]] == 'Positivo'][justificativas[i]].sum()
            negativos = df_filtered[df_filtered[sentimentos[i]] == 'Negativo'][justificativas[i]].sum()
            textos_positivos.extend(positivos)
            textos_negativos.extend(negativos)

        # Criar e mostrar as nuvens
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        if textos_positivos:
            wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(' '.join(textos_positivos))
            axes[0].imshow(wc_pos, interpolation='bilinear')
            axes[0].set_title('Principais Elogios', fontsize=20)
            axes[0].axis('off')
        else:
            axes[0].text(0.5, 0.5, 'Sem dados de elogios', horizontalalignment='center', verticalalignment='center', fontsize=15)
            axes[0].axis('off')

        if textos_negativos:
            wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(' '.join(textos_negativos))
            axes[1].imshow(wc_neg, interpolation='bilinear')
            axes[1].set_title('Principais Reclamações', fontsize=20)
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'Sem dados de reclamações', horizontalalignment='center', verticalalignment='center', fontsize=15, color='white')
            axes[1].axis('off')
            axes[1].set_facecolor('black')

        plt.tight_layout(pad=0)
        plt.show()

    analise_geral_satisfacao(df)
    analise_desempenho_aspectos(df)
    analise_correlacao_impacto(df)
    analise_temporal(df)
    analise_atendentes(df)
    analise_churn(df)
    nuvem_palavras(df)


