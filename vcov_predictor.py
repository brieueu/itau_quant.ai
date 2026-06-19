"""
Previsor de Matriz V-Cov com RNN - Módulo de Lógica de Negócio
Engenheiro de Software Sênior - Mercado Financeiro

Sistema avançado para previsão de matrizes de variância-covariância
usando decomposição de Cholesky e redes neurais LSTM.

Autor: Engenheiro de Software Sênior
Especialização: Python, TensorFlow/Keras, Análise Quantitativa
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.linalg import cholesky
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from alpha_weighting import AlphaWeighting
import warnings

warnings.filterwarnings('ignore')


class VCovPredictor:
    """
    Sistema de previsão de matrizes de variância-covariância usando RNN.

    Implementa metodologia avançada com decomposição de Cholesky e LSTM
    para previsão de estruturas de correlação em mercados financeiros.
    """

    def __init__(self):
        """Inicialização do sistema."""
        self.scalers = {}  # Scalers para cada série temporal
        self.model = None

    def predict_vcov_matrix(self, tickers_input, period, window, progress_callback=None):
        """
        Função principal para previsão da matriz V-Cov.
        
        Args:
            tickers_input (str): Tickers separados por vírgula
            period (int): Período histórico em anos
            window (int): Janela V-Cov em dias
            progress_callback (callable): Função callback para atualizações de progresso
        
        Returns:
            dict: Dicionário com resultado_texto, matriz_vcov, matriz_correlacao, tickers
        """
        try:
            # Validação dos inputs
            if not tickers_input.strip():
                raise ValueError("Insira pelo menos um ticker")
            
            if period < 1:
                raise ValueError("Período deve ser maior que 0")
                
            if window < 10:
                raise ValueError("Janela deve ser pelo menos 10 dias")

            # Processar tickers
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            
            # Etapas do processamento
            steps = [
                "📊 Baixando dados históricos...",
                "🔢 Calculando matrizes V-Cov históricas...",
                "🧮 Aplicando decomposição de Cholesky...",
                "📈 Criando séries temporais...",
                "🤖 Preparando dados para LSTM...",
                "🚀 Treinando modelo LSTM...",
                "🔮 Gerando previsão...",
                "✅ Previsão concluída!"
            ]
            
            # Etapa 1: Download dos dados
            if progress_callback:
                progress_callback(steps[0])
            price_data = self._download_data(tickers, period)
            
            # Etapa 2: Cálculo das matrizes históricas
            if progress_callback:
                progress_callback(steps[1])
            returns = price_data.pct_change().dropna()
            vcov_matrices = self._calculate_rolling_vcov(returns, window)

            # Etapa 3: Decomposição de Cholesky
            if progress_callback:
                progress_callback(steps[2])
            cholesky_matrices = self._apply_cholesky_decomposition(vcov_matrices)

            # Etapa 4: Criação das séries temporais
            if progress_callback:
                progress_callback(steps[3])
            time_series = self._create_time_series(cholesky_matrices)

            # Etapa 5: Preparação dos dados para LSTM
            if progress_callback:
                progress_callback(steps[4])
            X, y = self._prepare_lstm_data(time_series, sequence_length=60)

            # Etapa 6: Treinamento do modelo LSTM
            if progress_callback:
                progress_callback(steps[5])
            self.model = self._build_and_train_lstm(X, y, len(tickers))

            # Etapa 7: Previsão
            if progress_callback:
                progress_callback(steps[6])
            predicted_vcov = self._predict_next_vcov(time_series, len(tickers), steps_ahead=1)

            # Etapa 8: Finalizar
            if progress_callback:
                progress_callback(steps[7])
            
            # Gerar resultado textual
            result_text = self._generate_result_text(predicted_vcov, tickers)
            
            # Calcular matriz de correlação
            std_devs = np.sqrt(np.diag(predicted_vcov))
            correlation_matrix = predicted_vcov / np.outer(std_devs, std_devs)
            
            return {
                'success': True,
                'result_text': result_text,
                'vcov_matrix': predicted_vcov,
                'correlation_matrix': correlation_matrix,
                'tickers': tickers,
                'volatilities': [np.sqrt(predicted_vcov[i, i]) * 100 for i in range(len(tickers))]
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'result_text': f"❌ Erro durante o processamento: {str(e)}",
                'vcov_matrix': None,
                'correlation_matrix': None,
                'tickers': [],
                'volatilities': []
            }

    def _download_data(self, tickers, period):
        """Download dos dados históricos via yfinance."""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=period)

        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

            # Verificar se os dados foram baixados com sucesso
            if data is None or data.empty:
                raise ValueError("Nenhum dado foi baixado do yfinance")
            
            # Verificar se a coluna 'Close' existe
            if 'Close' not in data.columns.get_level_values(0):
                raise ValueError("Coluna 'Close' não encontrada nos dados baixados")

            # Tratar caso de ticker único
            if len(tickers) == 1:
                if isinstance(data['Close'], pd.Series):
                    prices = data['Close'].to_frame()
                    prices.columns = tickers
                else:
                    prices = data['Close']
            else:
                prices = data['Close']

            # Remover dados faltantes
            prices = prices.dropna()

            if prices.empty:
                raise ValueError("Nenhum dado válido encontrado para os tickers fornecidos")

            return prices
            
        except Exception as e:
            raise ValueError(f"Erro ao baixar dados: {str(e)}")

    def _calculate_rolling_vcov(self, returns, window):
        """Calcula matrizes de V-Cov em janela deslizante."""
        vcov_matrices = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window:i]
            vcov_matrix = window_returns.cov().values
            vcov_matrices.append(vcov_matrix)

        return np.array(vcov_matrices)

    def _apply_cholesky_decomposition(self, vcov_matrices):
        """Aplica decomposição de Cholesky nas matrizes V-Cov."""
        cholesky_matrices = []

        for vcov_matrix in vcov_matrices:
            try:
                # Adicionar pequena regularização se necessário
                vcov_matrix += np.eye(vcov_matrix.shape[0]) * 1e-6
                L = cholesky(vcov_matrix, lower=True)
                cholesky_matrices.append(L)
            except np.linalg.LinAlgError:
                # Se falhar, usar decomposição SVD como fallback
                U, s, Vt = np.linalg.svd(vcov_matrix)
                s = np.maximum(s, 1e-6)  # Regularização
                L = U @ np.diag(np.sqrt(s))
                cholesky_matrices.append(L)

        return np.array(cholesky_matrices)

    def _create_time_series(self, cholesky_matrices):
        """Extrai elementos da matriz triangular inferior como séries temporais."""
        n = cholesky_matrices.shape[1]  # Número de ativos
        n_series = n * (n + 1) // 2  # Número de séries temporais

        time_series = np.zeros((len(cholesky_matrices), n_series))

        for t, L in enumerate(cholesky_matrices):
            idx = 0
            for i in range(n):
                for j in range(i + 1):  # Apenas elementos da triangular inferior
                    time_series[t, idx] = L[i, j]
                    idx += 1

        return time_series

    def _prepare_lstm_data(self, time_series, sequence_length=60):
        """
        Prepara dados no formato adequado para LSTM com validação robusta.
        
        Args:
            time_series: Série temporal dos elementos da matriz de Cholesky
            sequence_length: Comprimento das sequências para o LSTM
            
        Returns:
            X, y: Dados preparados para treinamento
        """
        n_samples, n_features = time_series.shape
        
        print(f"📊 Preparando dados LSTM: {n_samples} amostras, {n_features} features")
        
        if n_samples < sequence_length + 50:  # Mínimo para treinar
            raise ValueError(f"Dados insuficientes: {n_samples} < {sequence_length + 50} (mínimo)")

        # Normalização robusta das séries temporais
        self.scalers = {}
        normalized_series = np.zeros_like(time_series)

        for i in range(n_features):
            # Usar StandardScaler para melhor estabilidade
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_series[:, i] = scaler.fit_transform(
                time_series[:, i].reshape(-1, 1)
            ).flatten()
            self.scalers[i] = scaler

        # Criação das sequências com overlap
        X, y = [], []
        step_size = 1  # Overlap total para mais dados de treino

        for i in range(sequence_length, n_samples, step_size):
            X.append(normalized_series[i - sequence_length:i])
            y.append(normalized_series[i])

        X, y = np.array(X), np.array(y)
        print(f"🔧 Sequências criadas: X{X.shape}, y{y.shape}")
        
        return X, y

    def _build_and_train_lstm(self, X, y, n_assets):
        """
        Constrói e treina o modelo LSTM com arquitetura otimizada para V-Cov.
        
        Args:
            X: Sequências de entrada
            y: Targets de saída
            n_assets: Número de ativos
            
        Returns:
            model: Modelo LSTM treinado
        """
        n_features = X.shape[2]
        sequence_length = X.shape[1]
        
        print(f"🏗️ Construindo LSTM: entrada={X.shape}, saída={y.shape}")

        # Arquitetura especializada para matrizes V-Cov
        model = keras.Sequential([
            # Primeira camada LSTM - captura padrões de longo prazo
            layers.LSTM(
                128, 
                return_sequences=True, 
                input_shape=(sequence_length, n_features),
                dropout=0.1,
                recurrent_dropout=0.1
            ),
            layers.BatchNormalization(),
            
            # Segunda camada LSTM - refina padrões
            layers.LSTM(
                64, 
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1
            ),
            layers.BatchNormalization(),
            
            # Terceira camada LSTM - extrai features finais
            layers.LSTM(
                32, 
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1
            ),
            
            # Camadas densas para reconstrução
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(n_features, activation='linear')  # Linear para valores contínuos
        ])

        # Compilação com otimizador otimizado
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            ),
            loss='huber',  # Huber loss é mais robusta para outliers
            metrics=['mae', 'mse']
        )

        # Split temporal (último 20% para validação)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"📚 Treinamento: {len(X_train)} amostras, Validação: {len(X_val)} amostras")

        # Callbacks para melhor treinamento
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Treinamento
        print("🚀 Iniciando treinamento LSTM...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Manter ordem temporal
        )
        
        # Avaliação final
        train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        print(f"✅ Treinamento concluído - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        return model

    def _predict_next_vcov(self, time_series, n_assets, steps_ahead=1):
        """
        Gera previsão da próxima matriz V-Cov garantindo positividade.
        
        Args:
            time_series: Série temporal completa
            n_assets: Número de ativos
            steps_ahead: Número de períodos à frente para prever
            
        Returns:
            vcov_pred_annual: Matriz V-Cov prevista anualizada
        """
        sequence_length = 60
        
        print(f"🔮 Gerando previsão para {steps_ahead} período(s) à frente...")

        # Preparar última sequência
        if len(time_series) < sequence_length:
            raise ValueError(f"Série muito curta: {len(time_series)} < {sequence_length}")
            
        last_sequence = time_series[-sequence_length:]

        # Normalizar usando os scalers treinados
        normalized_sequence = np.zeros_like(last_sequence)
        for i in range(last_sequence.shape[1]):
            normalized_sequence[:, i] = self.scalers[i].transform(
                last_sequence[:, i].reshape(-1, 1)
            ).flatten()

        # Predição iterativa para múltiplos steps
        current_sequence = normalized_sequence.copy()
        
        for step in range(steps_ahead):
            # Fazer previsão
            X_pred = current_sequence.reshape(1, sequence_length, -1)
            y_pred_normalized = self.model.predict(X_pred, verbose=0)[0]
            
            # Atualizar sequência para próxima predição
            if step < steps_ahead - 1:
                current_sequence = np.vstack([current_sequence[1:], y_pred_normalized])

        # Desnormalizar previsão final
        y_pred = np.zeros_like(y_pred_normalized)
        for i in range(len(y_pred_normalized)):
            y_pred[i] = self.scalers[i].inverse_transform(
                y_pred_normalized[i].reshape(-1, 1)
            )[0, 0]

        # Reconstruir matriz de Cholesky
        L_pred = self._reconstruct_cholesky_matrix(y_pred, n_assets)
        
        # Garantir que a matriz de Cholesky seja válida
        L_pred = self._ensure_valid_cholesky(L_pred)

        # Reconstruir matriz V-Cov
        vcov_pred = L_pred @ L_pred.T
        
        # Verificar se é positiva definida
        vcov_pred = self._ensure_positive_definite(vcov_pred)

        # Anualizar (252 dias úteis)
        vcov_pred_annual = vcov_pred * 252
        
        print(f"✅ Matriz V-Cov prevista - Shape: {vcov_pred_annual.shape}")
        print(f"📊 Determinante: {np.linalg.det(vcov_pred_annual):.2e}")
        print(f"📈 Eigenvalores mín/máx: {np.min(np.linalg.eigvals(vcov_pred_annual)):.2e}/{np.max(np.linalg.eigvals(vcov_pred_annual)):.2e}")

        return vcov_pred_annual
    
    def _reconstruct_cholesky_matrix(self, elements, n_assets):
        """Reconstrói matriz de Cholesky a partir dos elementos previstos."""
        L = np.zeros((n_assets, n_assets))
        idx = 0
        
        for i in range(n_assets):
            for j in range(i + 1):
                L[i, j] = elements[idx]
                idx += 1
                
        return L
    
    def _ensure_valid_cholesky(self, L):
        """Garante que a matriz de Cholesky seja válida."""
        n = L.shape[0]
        
        # Garantir que elementos diagonais sejam positivos
        for i in range(n):
            if L[i, i] <= 0:
                L[i, i] = max(abs(L[i, i]), 1e-6)
        
        # Zerar parte superior (garantir triangular inferior)
        for i in range(n):
            for j in range(i + 1, n):
                L[i, j] = 0
                
        return L
    
    def _ensure_positive_definite(self, matrix, min_eigenvalue=1e-8):
        """
        Garante que a matriz seja positiva definida.
        
        Args:
            matrix: Matriz a ser validada
            min_eigenvalue: Valor mínimo para eigenvalues
            
        Returns:
            matrix_pd: Matriz positiva definida
        """
        try:
            # Verificar se já é positiva definida
            eigvals = np.linalg.eigvals(matrix)
            
            if np.min(eigvals) > min_eigenvalue:
                return matrix  # Já é positiva definida
            
            # Se não, usar decomposição espectral para corrigir
            eigvals, eigvecs = np.linalg.eigh(matrix)
            
            # Corrigir eigenvalues negativos
            eigvals = np.maximum(eigvals, min_eigenvalue)
            
            # Reconstruir matriz
            matrix_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Verificar simetria
            matrix_pd = (matrix_pd + matrix_pd.T) / 2
            
            return matrix_pd
            
        except Exception as e:
            print(f"⚠️ Erro na validação de matriz positiva definida: {e}")
            # Fallback: adicionar regularização na diagonal
            return matrix + np.eye(matrix.shape[0]) * min_eigenvalue

    def _generate_result_text(self, predicted_vcov, tickers):
        """Gera o texto formatado do resultado."""
        result_text = "# 📊 MATRIZ DE VARIÂNCIA-COVARIÂNCIA PREVISTA (ANUALIZADA)\n\n"
        
        # Informações gerais
        result_text += f"**Ativos:** {', '.join(tickers)}\n"
        result_text += f"**Data da previsão:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Matriz formatada
        result_text += "## 📈 Matriz V-Cov (valores em formato científico):\n\n"
        
        # Criar tabela markdown
        result_text += "| Ativo |"
        for ticker in tickers:
            result_text += f" {ticker} |"
        result_text += "\n|-------|"
        for _ in tickers:
            result_text += "------|"
        result_text += "\n"
        
        # Linhas da matriz
        for i, ticker in enumerate(tickers):
            result_text += f"| **{ticker}** |"
            for j in range(len(tickers)):
                result_text += f" {predicted_vcov[i, j]:.4e} |"
            result_text += "\n"

        # Volatilidades (diagonal)
        result_text += "\n## 📊 Volatilidades Anualizadas:\n\n"
        for i, ticker in enumerate(tickers):
            vol = np.sqrt(predicted_vcov[i, i]) * 100
            result_text += f"- **{ticker}**: {vol:.2f}%\n"

        return result_text

    def calculate_alpha_weighted_portfolio(self, tickers_input, benchmark="^GSPC", risk_free_rate=0.02, period="2y"):
        """
        Calcula ponderação alfa para o portfólio.
        
        Args:
            tickers_input (str): Tickers separados por vírgula
            benchmark (str): Benchmark para cálculo do alfa
            risk_free_rate (float): Taxa livre de risco anual
            period (str): Período para análise
            
        Returns:
            dict: Resultado da análise alfa com pesos e estatísticas
        """
        try:
            # Processar tickers
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            
            if len(tickers) < 2:
                raise ValueError("Insira pelo menos 2 tickers para análise alfa")
            
            # Criar calculador alfa
            alpha_calc = AlphaWeighting(benchmark=benchmark, risk_free_rate=risk_free_rate)
            
            # Calcular pesos
            weights = alpha_calc.calculate_portfolio_weights(tickers, period)
            
            # Obter estatísticas
            stats = alpha_calc.get_portfolio_stats()
            
            # Preparar resultado
            result = {
                'weights': weights,
                'alphas': alpha_calc.alphas,
                'betas': alpha_calc.betas,
                'stats': stats,
                'tickers': tickers
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f"Erro na análise alfa: {str(e)}",
                'weights': {ticker: 1.0/len(tickers) for ticker in tickers} if 'tickers' in locals() else {},
                'alphas': {},
                'betas': {},
                'stats': {},
                'tickers': []
            }

    def format_alpha_results(self, alpha_result):
        """
        Formata resultados da análise alfa para exibição.
        
        Args:
            alpha_result (dict): Resultado da análise alfa
            
        Returns:
            str: Texto formatado com resultados
        """
        if 'error' in alpha_result:
            return f"❌ {alpha_result['error']}"
        
        result_text = "# 🎯 ANÁLISE DE PONDERAÇÃO ALFA\n\n"
        
        # Informações gerais
        result_text += f"**Data da análise:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result_text += f"**Ativos analisados:** {', '.join(alpha_result['tickers'])}\n\n"
        
        # Estatísticas do portfólio
        stats = alpha_result['stats']
        if stats:
            result_text += "## 📊 Estatísticas do Portfólio:\n\n"
            result_text += f"- **Alfa Total:** {stats.get('total_alpha', 0):.4f}\n"
            result_text += f"- **Beta Médio:** {stats.get('average_beta', 0):.4f}\n"
            result_text += f"- **Número de Ativos:** {stats.get('num_assets', 0)}\n"
            result_text += f"- **Concentração Máxima:** {stats.get('alpha_concentration', 0):.2%}\n\n"
        
        # Detalhes por ativo
        result_text += "## 📈 Análise por Ativo:\n\n"
        result_text += "| Ativo | Peso | Alfa | Beta | Status |\n"
        result_text += "|-------|------|------|------|--------|\n"
        
        alphas = alpha_result['alphas']
        betas = alpha_result['betas']
        weights = alpha_result['weights']
        
        for ticker in alpha_result['tickers']:
            alpha = alphas.get(ticker, 0)
            beta = betas.get(ticker, 0)
            weight = weights.get(ticker, 0)
            status = "🟢 Outperform" if alpha > 0 else "🔴 Underperform"
            
            result_text += f"| **{ticker}** | {weight:.2%} | {alpha:.4f} | {beta:.4f} | {status} |\n"
        
        # Interpretação
        result_text += "\n## 🔍 Interpretação:\n\n"
        
        positive_alpha_count = sum(1 for alpha in alphas.values() if alpha > 0)
        total_assets = len(alphas)
        
        if positive_alpha_count > total_assets / 2:
            result_text += "✅ **Portfólio promissor**: Maioria dos ativos apresenta alfa positivo.\n\n"
        else:
            result_text += "⚠️ **Portfólio defensivo**: Maioria dos ativos apresenta alfa negativo ou neutro.\n\n"
        
        max_weight_ticker = max(weights.keys(), key=lambda x: weights[x]) if weights else None
        if max_weight_ticker:
            result_text += f"🎯 **Maior concentração**: {max_weight_ticker} ({weights[max_weight_ticker]:.2%})\n\n"
        
        # Recomendações
        result_text += "## 💡 Recomendações:\n\n"
        result_text += "- Use esta ponderação para estratégias long-short\n"
        result_text += "- Monitore regularmente os alfas (podem mudar no tempo)\n"
        result_text += "- Considere rebalanceamento trimestral\n"
        result_text += "- Ativos com alfa negativo podem ser candidatos a short\n\n"
        
        return result_text
