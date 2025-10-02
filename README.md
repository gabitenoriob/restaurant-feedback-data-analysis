# Análise de Dados de Feedback de Restaurante

## 📝 Descrição

Este projeto realiza uma análise exploratória dos dados de feedback de clientes de um restaurante. O objetivo é extrair insights valiosos a partir das avaliações, comentários e notas deixadas pelos clientes, a fim de identificar pontos fortes, áreas de melhoria e padrões no serviço oferecido.

## 🎯 Objetivos da Análise

A análise busca responder a perguntas como:

- Qual é a distribuição geral das notas de avaliação?
- Quais são os pratos ou aspectos do serviço mais elogiados e mais criticados?
- Existe alguma tendência ou padrão nos feedbacks ao longo do tempo?
- Qual é o sentimento geral (positivo, negativo, neutro) expresso nos comentários?
- Há alguma correlação entre a nota da avaliação e o tamanho do comentário?

## 📊 Fonte de Dados

O conjunto de dados utilizado neste projeto foi obtido de feedbacks reais de um restaurante em Maceió-AL. Ele contém as seguintes colunas:

- **id_avaliacao:** Identificador único para cada feedback.
- **nota:** A nota dada pelo cliente (ex: de 1 a 5).
- **comentario:** O texto do feedback deixado pelo cliente.
- **data:** A data em que o feedback foi registrado.
- etc

## 🛠️ Tecnologias Utilizadas

Este projeto foi desenvolvido utilizando as seguintes tecnologias e bibliotecas:

- **Linguagem:** Python 3.11
- **Bibliotecas de Análise:**
  - `pandas`: Para manipulação e limpeza dos dados.
  - `numpy`: Para operações numéricas.
- **Bibliotecas de Visualização:**
  - `matplotlib`: Para a criação de gráficos estáticos.
  - `seaborn`: Para visualizações estatísticas mais atraentes.
  - `wordcloud`: Para criar nuvens de palavras a partir dos comentários.
- **Ambiente de Desenvolvimento:** Jupyter Notebook

## 🚀 Como Utilizar

Para replicar esta análise, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/gabitenoriob/restaurant-feedback-data-analysis.git](https://github.com/gabitenoriob/restaurant-feedback-data-analysis.git)
    ```

2.  **Navegue até o diretório do projeto:**
    ```bash
    cd restaurant-feedback-data-analysis
    ```

3.  **Instale as dependências necessárias:**
    ```bash
    pip install pandas numpy matplotlib seaborn wordcloud jupyter
    ```

4.  **Inicie o Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

5.  Abra o arquivo `analise_feedback.ipynb` e execute as células de código.

## 👤 Autor

**Gabriela Tenório, Beatriz Motta e Frederico Filho**

- **GitHub:** [@gabitenoriob](https://github.com/gabitenoriob) 
- **LinkedIn:** []
