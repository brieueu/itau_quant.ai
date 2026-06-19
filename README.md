# Itaú Quant AI

Projeto educacional de análise quantitativa para estimar risco, correlação e pesos de portfólio a partir de dados históricos de mercado.

A aplicação combina três ideias principais:

- previsão de matriz de variância-covariância (V-Cov) com LSTM;
- cálculo de alfa e beta via CAPM;
- otimização de portfólio com métricas de risco e retorno.

> Aviso: este projeto é para estudo e pesquisa. Não é recomendação de investimento.

## Fonte dos dados

Os dados são baixados automaticamente pelo pacote `yfinance`, que consulta dados públicos disponibilizados pelo Yahoo Finance.

O projeto usa principalmente:

- preços históricos de fechamento ajustado dos ativos informados pelo usuário;
- benchmark de mercado, por padrão `^GSPC` (S&P 500);
- retornos calculados a partir desses preços.

Os dados não ficam armazenados no repositório. Eles são coletados em tempo de execução conforme os tickers, o período e a janela escolhidos na interface ou nos scripts.

Exemplos de tickers aceitos:

```text
AAPL, GOOGL, MSFT, AMZN, TSLA
```

## Como funciona

### 1. Entrada dos ativos

O usuário informa uma lista de tickers e parâmetros como período histórico, janela da matriz V-Cov, benchmark e taxa livre de risco.

### 2. Coleta e preparação dos dados

O sistema baixa os preços históricos pelo `yfinance`, remove dados ausentes e calcula os retornos dos ativos.

### 3. Matriz V-Cov histórica

Com os retornos, o sistema calcula matrizes de variância-covariância em janelas móveis. Essas matrizes representam o risco individual dos ativos e a relação de movimento entre eles.

### 4. Decomposição de Cholesky

Cada matriz V-Cov é transformada por decomposição de Cholesky. Isso ajuda a manter a consistência matemática da matriz e transforma seus elementos em séries temporais que podem ser usadas pela rede neural.

### 5. Previsão com LSTM

As séries temporais extraídas das matrizes são usadas para treinar uma rede LSTM. O modelo aprende padrões recentes de volatilidade e correlação e gera uma previsão para a próxima matriz V-Cov.

### 6. Alfa, beta e pesos

O módulo de ponderação alfa calcula, para cada ativo:

- beta: sensibilidade em relação ao benchmark;
- alfa: retorno excedente estimado pelo CAPM;
- peso sugerido: maior para ativos com melhor alfa ajustado.

Quando o modelo híbrido é usado, a previsão de risco da matriz V-Cov é combinada com os retornos esperados por CAPM + alfa para otimizar o portfólio.

### 7. Otimização e backtest

O projeto também possui módulos para testar estratégias como:

- maximização de Sharpe;
- mínima variância;
- máximo retorno;
- risk parity;
- ponderação por alfa.

O backtest simula rebalanceamentos periódicos e calcula métricas como retorno total, volatilidade, Sharpe, drawdown, VaR e CVaR.

## Estrutura principal

```text
.
├── main.py                    # Ponto de entrada da aplicação Gradio
├── vcov_predictor.py          # Previsão da matriz V-Cov com LSTM
├── alpha_weighting.py         # Cálculo de alfa, beta e pesos por CAPM
├── hybrid_portfolio_model.py  # Combina V-Cov, alfa e otimização de portfólio
├── backtest_engine.py         # Backtest das estratégias
├── analise_lucros.py          # Script simples de comparação de lucros
├── interface/                 # Interfaces Gradio
├── requirements.txt           # Dependências Python
└── README.md
```

## Como executar

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Execute a aplicação:

```bash
python main.py
```

Depois acesse no navegador:

```text
http://localhost:7870
```

## Tecnologias usadas

- Python
- Pandas e NumPy
- SciPy
- scikit-learn
- TensorFlow/Keras
- yfinance
- Gradio
- Plotly

## Limitações

- A qualidade dos resultados depende da disponibilidade e consistência dos dados do Yahoo Finance.
- O modelo usa dados históricos; isso não garante comportamento futuro do mercado.
- Custos, liquidez, impostos e restrições reais de negociação podem não estar totalmente representados.
- O projeto tem finalidade educacional e deve ser validado antes de qualquer uso prático.
