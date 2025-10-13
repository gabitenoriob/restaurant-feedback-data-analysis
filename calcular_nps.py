import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def calcular_nps(df):
    # Features numéricas (exceto a recomendação)
    features = ['service_rating', 'food_rating', 'environment_rating']  
    X = df[features]
    y = df['recommendation_rating']

    # Treinar o modelo de regressão linear
    model = LinearRegression()
    model.fit(X, y)

    # Exibir os coeficientes
    print("Impacto de cada aspecto na nota de recomendação:")
    for feature, coef in zip(features, model.coef_):
        print(f"Cada ponto a mais em {feature} aumenta {coef:.2f} pontos na recomendação.")

    plt.bar(features, model.coef_)
    plt.ylabel('Impacto na recomendação')
    plt.title('Coeficientes da Regressão Linear')
    plt.show()

    df['nps_pred'] = model.predict(X)
    df['recommendation_rating'] = df['recommendation_rating'].fillna(df['nps_pred'])

    return df
