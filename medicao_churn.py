from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def medicao_churn(df):
    # Selecionar features relevantes
    features = [
        'service_rating', 'food_rating', 'environment_rating',
        'sentimento_comida', 'sentimento_ambiente', 'sentimento_servico',  
        'categoria_nps', 
        'nome_topico', 'probabilidade_topico'  
    ]

    target = 'churn'

    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    for feature, importance in zip(X.columns, rf.feature_importances_):
        print(f"Feature: {feature}, Importance: {importance:.4f}")

    # Predizer churn para todo o df
    df_pred = df.copy()
    X_all = pd.get_dummies(df_pred[features], drop_first=True)
    X_all = X_all.reindex(columns=X.columns, fill_value=0)
    df_pred['churn_pred'] = rf.predict(X_all)
    df_pred['churn_prob'] = rf.predict_proba(X_all)[:, 1]

    return df_pred
