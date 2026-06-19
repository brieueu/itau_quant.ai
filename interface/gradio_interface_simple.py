"""
Interface Gradio para o Previsor de Matriz V-Cov com RNN
Engenheiro de Software Sênior - Mercado Financeiro

Interface web moderna usando Gradio para interação com o sistema de 
previsão de matrizes de variância-covariância.

Autor: Engenheiro de Software Sênior
Especialização: Python, TensorFlow/Keras, Análise Quantitativa
"""

import gradio as gr
import plotly.graph_objects as go
from vcov_predictor import VCovPredictor
from hybrid_portfolio_model import HybridPortfolioModel, PortfolioConstraints
from backtest_engine import BacktestEngine, BacktestConfig, run_comprehensive_backtest


class GradioInterface:
    """Interface Gradio para o sistema de previsão V-Cov."""
    
    def __init__(self):
        """Inicialização da interface."""
        self.predictor = VCovPredictor()
        self.hybrid_model = HybridPortfolioModel()
    
    def predict_wrapper(self, tickers, period, window):
        """
        Wrapper da função de previsão para Gradio.
        
        Args:
            tickers (str): Tickers separados por vírgula
            period (int): Período histórico em anos  
            window (int): Janela V-Cov em dias
            
        Yields:
            tuple: (heatmap_vcov, heatmap_corr, resultado_markdown)
        """
        # Executar previsão
        result = self.predictor.predict_vcov_matrix(
            tickers, int(period), int(window)
        )
        
        if not result['success']:
            # Em caso de erro - retornar None para os gráficos e erro no texto
            return None, None, result['result_text']
        
        # Sucesso - criar visualizações
        vcov_fig = self._create_vcov_heatmap(
            result['vcov_matrix'], 
            result['tickers']
        )
        
        corr_fig = self._create_correlation_heatmap(
            result['correlation_matrix'], 
            result['tickers']
        )
        
        # Resultado final
        return vcov_fig, corr_fig, result['result_text']
    

    
    def alpha_weighting_wrapper(self, tickers_input, benchmark, risk_free_rate):
        """Wrapper para análise de ponderação alfa."""
        try:
            # Converter taxa de juros para float
            risk_free_rate = float(risk_free_rate) / 100 if risk_free_rate else 0.02
            
            # Realizar análise alfa
            alpha_result = self.predictor.calculate_alpha_weighted_portfolio(
                tickers_input=tickers_input,
                benchmark=benchmark,
                risk_free_rate=risk_free_rate
            )
            
            # Formatar resultado
            alpha_text = self.predictor.format_alpha_results(alpha_result)
            
            # Criar gráfico de pesos
            weights_fig = self._create_weights_chart(alpha_result)
            
            # Criar gráfico alfa vs beta
            alpha_beta_fig = self._create_alpha_beta_scatter(alpha_result)
            
            return weights_fig, alpha_beta_fig, alpha_text
            
        except Exception as e:
            error_msg = f"❌ **Erro na análise alfa**: {str(e)}"
            empty_fig = go.Figure()
            return empty_fig, empty_fig, error_msg
    
    def hybrid_portfolio_wrapper(
        self, 
        tickers, 
        vcov_period, 
        vcov_window, 
        alpha_period,
        max_weight,
        min_weight,
        optimization_method,
        allow_short
    ):
        """Wrapper para o modelo híbrido de portfólio."""
        try:
            # Configurar restrições
            constraints = PortfolioConstraints(
                max_weight=float(max_weight) / 100,
                min_weight=float(min_weight) / 100,
                allow_short=allow_short
            )
            
            # Executar análise híbrida
            results = self.hybrid_model.generate_portfolio_recommendation(
                tickers=tickers,
                vcov_period=int(vcov_period),
                vcov_window=int(vcov_window),
                alpha_period=alpha_period,
                constraints=constraints,
                optimization_method=optimization_method
            )
            
            # Gerar relatório
            report = self.hybrid_model.generate_detailed_report()
            
            # Criar visualizações
            weights_fig = self._create_hybrid_weights_chart(results)
            metrics_fig = self._create_portfolio_metrics_chart(results)
            
            return weights_fig, metrics_fig, report
            
        except Exception as e:
            error_msg = f"❌ **Erro na análise híbrida**: {str(e)}"
            empty_fig = go.Figure()
            return empty_fig, empty_fig, error_msg
    
    def backtest_wrapper(self, tickers, start_date, end_date, rebalance_freq, initial_capital):
        """Wrapper para executar backtest completo."""
        try:
            print(f"🚀 Iniciando backtest para: {tickers}")
            
            # Processar entrada
            tickers_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
            
            if len(tickers_list) < 2:
                return None, None, "❌ **Erro**: Insira pelo menos 2 tickers para backtest"
            
            # Executar backtest
            results = run_comprehensive_backtest(
                tickers=tickers_list,
                start_date=start_date,
                end_date=end_date,
                rebalance_frequency=int(rebalance_freq),
                initial_capital=float(initial_capital)
            )
            
            if not results:
                return None, None, "❌ **Erro**: Falha na execução do backtest"
            
            # Gerar relatório
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                rebalance_frequency=int(rebalance_freq),
                initial_capital=float(initial_capital)
            )
            engine = BacktestEngine(config)
            report = engine.generate_comparison_report(results)
            
            # Criar visualizações
            performance_fig = self._create_backtest_performance_chart(results)
            metrics_fig = self._create_backtest_metrics_chart(results)
            
            return performance_fig, metrics_fig, report
            
        except Exception as e:
            error_msg = f"❌ **Erro no backtest**: {str(e)}"
            empty_fig = go.Figure()
            return empty_fig, empty_fig, error_msg
    
    def _create_hybrid_weights_chart(self, results):
        """Cria gráfico de pesos otimizados do modelo híbrido."""
        tickers = list(results.optimal_weights.keys())
        weights = [results.optimal_weights[ticker] * 100 for ticker in tickers]
        alphas = [results.alpha_contribution[ticker] for ticker in tickers]
        
        # Cores baseadas no alfa
        colors = ['green' if alpha > 0 else 'red' if alpha < 0 else 'gray' for alpha in alphas]
        
        fig = go.Figure(data=[
            go.Bar(
                x=tickers,
                y=weights,
                marker_color=colors,
                text=[f"{w:.1f}%" for w in weights],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>" +
                             "Peso: %{y:.1f}%<br>" +
                             "Alfa: %{customdata:.4f}<extra></extra>",
                customdata=alphas
            )
        ])
        
        fig.update_layout(
            title="🎯 Pesos Otimizados do Portfólio Híbrido",
            xaxis_title="Ativos",
            yaxis_title="Peso (%)",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_portfolio_metrics_chart(self, results):
        """Cria gráfico radar com métricas do portfólio."""
        metrics = [
            'Sharpe Ratio',
            'Retorno Esperado (%)',
            'Diversificação',
            'Contribuição Alfa'
        ]
        
        values = [
            min(results.sharpe_ratio * 20, 100),  # Normalizar Sharpe
            results.expected_return * 100,
            min(results.diversification_ratio * 30, 100),  # Normalizar diversificação
            sum(results.alpha_contribution.values()) * 1000  # Normalizar alfa
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Portfólio Otimizado',
            line_color='rgb(106, 81, 163)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="📊 Métricas do Portfólio Híbrido",
            height=400
        )
        
        return fig
    
    def _create_weights_chart(self, alpha_result):
        """Cria gráfico de barras com os pesos calculados."""
        if 'error' in alpha_result or not alpha_result['weights']:
            return go.Figure()
        
        tickers = list(alpha_result['weights'].keys())
        weights = list(alpha_result['weights'].values())
        alphas = [alpha_result['alphas'].get(t, 0) for t in tickers]
        
        # Cores baseadas no alfa
        colors = ['green' if alpha > 0 else 'red' for alpha in alphas]
        
        fig = go.Figure(data=go.Bar(
            x=tickers,
            y=[w * 100 for w in weights],  # Converter para porcentagem
            marker_color=colors,
            text=[f"{w:.1%}" for w in weights],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="🎯 Ponderação Alfa - Pesos do Portfólio",
            xaxis_title="Ativos",
            yaxis_title="Peso (%)",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_alpha_beta_scatter(self, alpha_result):
        """Cria gráfico de dispersão alfa vs beta."""
        if 'error' in alpha_result or not alpha_result['alphas']:
            return go.Figure()
        
        tickers = list(alpha_result['alphas'].keys())
        alphas = list(alpha_result['alphas'].values())
        betas = [alpha_result['betas'].get(t, 0) for t in tickers]
        weights = [alpha_result['weights'].get(t, 0) for t in tickers]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=betas,
            y=alphas,
            mode='markers+text',
            marker=dict(
                size=[w * 1000 for w in weights],  # Tamanho proporcional ao peso
                color=alphas,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Alfa")
            ),
            text=tickers,
            textposition="middle center",
            hovertemplate="<b>%{text}</b><br>" +
                         "Beta: %{x:.3f}<br>" +
                         "Alfa: %{y:.3f}<br>" +
                         "<extra></extra>"
        ))
        
        # Linhas de referência
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="📊 Análise Alfa vs Beta",
            xaxis_title="Beta (Sensibilidade ao Mercado)",
            yaxis_title="Alfa (Retorno em Excesso)",
            height=500,
            annotations=[
                dict(x=0.02, y=0.98, xref="paper", yref="paper",
                     text="Tamanho da bolha = Peso no portfólio", 
                     showarrow=False, font=dict(size=10))
            ]
        )
        
        return fig
    
    def _create_backtest_performance_chart(self, results):
        """Cria gráfico de performance das estratégias no backtest."""
        fig = go.Figure()
        
        # Ordenar estratégias por Sharpe ratio
        strategies = [name for name in results.keys() if name != "benchmark"]
        strategies.sort(key=lambda x: results[x].sharpe_ratio, reverse=True)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, strategy in enumerate(strategies):
            metrics = results[strategy]
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Bar(
                name=strategy.upper(),
                x=['Retorno Total', 'Retorno Anual', 'Sharpe Ratio', 'Calmar Ratio'],
                y=[
                    metrics.total_return * 100,
                    metrics.annualized_return * 100,
                    metrics.sharpe_ratio * 10,  # Multiplicar para visualização
                    metrics.calmar_ratio * 10   # Multiplicar para visualização
                ],
                marker_color=color,
                hovertemplate="<b>%{fullData.name}</b><br>" +
                             "%{x}: %{y:.2f}<br>" +
                             "<extra></extra>"
            ))
        
        # Adicionar benchmark se disponível
        if "benchmark" in results:
            benchmark = results["benchmark"]
            fig.add_trace(go.Bar(
                name="BENCHMARK",
                x=['Retorno Total', 'Retorno Anual', 'Sharpe Ratio', 'Calmar Ratio'],
                y=[
                    benchmark.total_return * 100,
                    benchmark.annualized_return * 100,
                    benchmark.sharpe_ratio * 10,
                    benchmark.calmar_ratio * 10
                ],
                marker_color='lightgray',
                opacity=0.7
            ))
        
        fig.update_layout(
            title="📊 Comparação de Performance - Backtest",
            xaxis_title="Métricas",
            yaxis_title="Valores (%)",
            barmode='group',
            height=500,
            annotations=[
                dict(x=0.02, y=0.98, xref="paper", yref="paper",
                     text="*Sharpe e Calmar multiplicados por 10 para visualização", 
                     showarrow=False, font=dict(size=10))
            ]
        )
        
        return fig
    
    def _create_backtest_metrics_chart(self, results):
        """Cria gráfico radar comparando métricas de risco-retorno."""
        strategies = [name for name in results.keys() if name != "benchmark"]
        
        fig = go.Figure()
        
        colors = ['rgba(31, 119, 180, 0.7)', 'rgba(255, 127, 14, 0.7)', 
                  'rgba(44, 160, 44, 0.7)', 'rgba(214, 39, 40, 0.7)', 
                  'rgba(148, 103, 189, 0.7)']
        
        for i, strategy in enumerate(strategies):
            metrics = results[strategy]
            
            # Normalizar métricas para escala 0-100
            values = [
                min(metrics.sharpe_ratio * 20, 100),  # Sharpe ratio
                max(0, 100 + metrics.max_drawdown * 200),  # Max drawdown (invertido)
                min(metrics.volatility * 500, 100),  # Volatilidade (invertida)
                min(metrics.win_rate * 100, 100),  # Win rate
                min(abs(metrics.alpha) * 500, 100),  # Alpha
                min(metrics.calmar_ratio * 20, 100)  # Calmar ratio
            ]
            
            categories = [
                'Sharpe Ratio',
                'Controle DD',
                'Baixa Volatilidade',
                'Taxa Acerto',
                'Alpha',
                'Calmar Ratio'
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=strategy.upper(),
                line=dict(color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)].replace('0.7', '0.3')
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                )
            ),
            title="🎯 Comparação Multi-dimensional de Estratégias",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_vcov_heatmap(self, predicted_vcov, tickers):
        """Cria heatmap da matriz de variância-covariância."""
        fig = go.Figure(data=go.Heatmap(
            z=predicted_vcov,
            x=tickers,
            y=tickers,
            colorscale='RdYlBu_r',
            showscale=True,
            text=[[f"{predicted_vcov[i,j]:.2e}" for j in range(len(tickers))] 
                  for i in range(len(tickers))],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🔥 Matriz de Variância-Covariância (Anualizada)",
            xaxis_title="Ativos",
            yaxis_title="Ativos",
            font=dict(size=12),
            height=500,
            width=600
        )
        
        return fig

    def _create_correlation_heatmap(self, corr_matrix, tickers):
        """Cria heatmap da matriz de correlação."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=tickers,
            y=tickers,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=True,
            text=[[f"{corr_matrix[i,j]:.3f}" for j in range(len(tickers))] 
                  for i in range(len(tickers))],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🎯 Matriz de Correlações",
            xaxis_title="Ativos",
            yaxis_title="Ativos",
            font=dict(size=12),
            height=500,
            width=600
        )
        
        return fig
    
    def create_interface(self):
        """Cria a interface Gradio completa."""
        
        with gr.Blocks(
            title="Previsor de Matriz V-Cov com RNN",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
            }
            """
        ) as demo:
            
            gr.Markdown("# 🚀 Previsor de Matriz V-Cov com RNN + Ponderação Alfa")
            
            with gr.Tabs():
                with gr.TabItem("📈 Predição V-Cov"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Configurações V-Cov")
                            
                            tickers_input = gr.Textbox(
                                label="Tickers (separados por vírgula)",
                                value="PETR4.SA, VALE3.SA, ITUB4.SA",
                                placeholder="Ex: AAPL, GOOGL, MSFT",
                                info="Digite os códigos dos ativos financeiros separados por vírgula"
                            )
                            
                            period_input = gr.Textbox(
                                label="Período histórico (anos)",
                                value="5",
                                placeholder="Ex: 5",
                                info="Quantidade de anos de dados históricos para análise"
                            )
                            
                            window_input = gr.Textbox(
                                label="Janela V-Cov (dias)",
                                value="90",
                                placeholder="Ex: 90",
                                info="Janela deslizante para cálculo das matrizes de variância-covariância"
                            )
                            
                            predict_btn = gr.Button(
                                "🔮 Gerar Predição V-Cov",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("## Resultados V-Cov")
                            
                            with gr.Tabs():
                                with gr.TabItem("📋 Relatório Detalhado"):
                                    result_markdown = gr.Markdown(
                                        value="Aguardando previsão...",
                                        height=400
                                    )
                                
                                with gr.TabItem("🔥 Matriz V-Cov"):
                                    vcov_plot = gr.Plot(
                                        label="Heatmap da Matriz de Variância-Covariância"
                                    )
                                
                                with gr.TabItem("🎯 Correlações"):
                                    corr_plot = gr.Plot(
                                        label="Heatmap da Matriz de Correlações"
                                    )
                
                with gr.TabItem("🎯 Ponderação Alfa"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Configurações Ponderação Alfa")
                            
                            alpha_tickers_input = gr.Textbox(
                                label="Tickers para análise alfa",
                                value="AAPL, GOOGL, MSFT, AMZN, TSLA",
                                placeholder="Ex: AAPL, GOOGL, MSFT",
                                info="Ativos para calcular ponderação baseada em alfa"
                            )
                            
                            benchmark_input = gr.Textbox(
                                label="Benchmark",
                                value="^GSPC",
                                placeholder="^GSPC (S&P 500)",
                                info="Índice de referência para cálculo do alfa"
                            )
                            
                            risk_free_input = gr.Textbox(
                                label="Taxa livre de risco (%)",
                                value="2.0",
                                placeholder="2.0",
                                info="Taxa livre de risco anual em porcentagem"
                            )
                            
                            alpha_btn = gr.Button(
                                "🎯 Calcular Ponderação Alfa",
                                variant="secondary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("## Resultados Ponderação Alfa")
                            
                            with gr.Tabs():
                                with gr.TabItem("📋 Análise Detalhada"):
                                    alpha_result_markdown = gr.Markdown(
                                        value="Aguardando análise alfa...",
                                        height=400
                                    )
                                
                                with gr.TabItem("⚖️ Pesos do Portfólio"):
                                    weights_plot = gr.Plot(
                                        label="Ponderação dos Ativos"
                                    )
                                
                                with gr.TabItem("📊 Alfa vs Beta"):
                                    alpha_beta_plot = gr.Plot(
                                        label="Dispersão Alfa vs Beta"
                                    )
                
                with gr.TabItem("🚀 Modelo Híbrido V-Cov + Alfa"):
                    gr.Markdown("## 🧠 Otimização Avançada com IA")
                    gr.Markdown("*Combina previsão LSTM de V-Cov com análise alfa para otimização quantitativa de portfólio*")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Parâmetros de entrada
                            hybrid_tickers_input = gr.Textbox(
                                label="Tickers para otimização",
                                value="AAPL, GOOGL, MSFT, AMZN, TSLA",
                                placeholder="Ex: AAPL, GOOGL, MSFT, AMZN"
                            )
                            
                            with gr.Row():
                                hybrid_vcov_period = gr.Number(
                                    label="Período V-Cov (anos)",
                                    value=2,
                                    minimum=1,
                                    maximum=10
                                )
                                
                                hybrid_vcov_window = gr.Number(
                                    label="Janela V-Cov (dias)",
                                    value=60,
                                    minimum=30,
                                    maximum=252
                                )
                            
                            hybrid_alpha_period = gr.Dropdown(
                                label="Período Alfa",
                                choices=["1y", "2y", "3y", "5y"],
                                value="2y"
                            )
                            
                            # Restrições do portfólio
                            gr.Markdown("### ⚙️ Restrições")
                            
                            with gr.Row():
                                hybrid_max_weight = gr.Number(
                                    label="Peso máximo (%)",
                                    value=30,
                                    minimum=5,
                                    maximum=100
                                )
                                
                                hybrid_min_weight = gr.Number(
                                    label="Peso mínimo (%)",
                                    value=5,
                                    minimum=0,
                                    maximum=50
                                )
                            
                            hybrid_optimization_method = gr.Dropdown(
                                label="Método de otimização",
                                choices=[
                                    "max_sharpe",
                                    "min_variance", 
                                    "max_return",
                                    "risk_parity",
                                    "alpha_weighted"
                                ],
                                value="max_sharpe"
                            )
                            
                            hybrid_allow_short = gr.Checkbox(
                                label="Permitir posições vendidas",
                                value=False
                            )
                            
                            hybrid_btn = gr.Button(
                                "🚀 Otimizar Portfólio Híbrido",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.TabItem("📋 Relatório Completo"):
                                    hybrid_report_output = gr.Markdown(
                                        value="Configure os parâmetros e clique em 'Otimizar Portfólio Híbrido'...",
                                        height=600
                                    )
                                
                                with gr.TabItem("🎯 Pesos Otimizados"):
                                    hybrid_weights_plot = gr.Plot(
                                        label="Alocação Otimizada"
                                    )
                                
                                with gr.TabItem("📊 Métricas do Portfólio"):
                                    hybrid_metrics_plot = gr.Plot(
                                        label="Radar de Performance"
                                    )
                
                with gr.TabItem("🧪 Backtest Comparativo"):
                    gr.Markdown("## 📊 Teste Histórico de Todas as Estratégias")
                    gr.Markdown("*Execute backtest comparativo de todas as estratégias de otimização com métricas detalhadas*")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎯 Configurações do Backtest")
                            
                            backtest_tickers_input = gr.Textbox(
                                label="Tickers para backtest",
                                value="AAPL, GOOGL, MSFT, AMZN, TSLA",
                                placeholder="Ex: AAPL, GOOGL, MSFT, AMZN",
                                info="Ativos para análise comparativa"
                            )
                            
                            with gr.Row():
                                backtest_start_date = gr.Textbox(
                                    label="Data inicial",
                                    value="2020-01-01",
                                    placeholder="YYYY-MM-DD"
                                )
                                
                                backtest_end_date = gr.Textbox(
                                    label="Data final", 
                                    value="2024-01-01",
                                    placeholder="YYYY-MM-DD"
                                )
                            
                            with gr.Row():
                                backtest_rebalance_freq = gr.Number(
                                    label="Frequência rebalanceamento (dias)",
                                    value=30,
                                    minimum=1,
                                    maximum=365,
                                    step=1
                                )
                                
                                backtest_initial_capital = gr.Number(
                                    label="Capital inicial ($)",
                                    value=100000,
                                    minimum=1000,
                                    maximum=10000000,
                                    step=1000
                                )
                            
                            gr.Markdown("### 📈 Estratégias Testadas")
                            gr.Markdown("""
                            - **Max Sharpe**: Maximiza relação risco-retorno
                            - **Min Variance**: Minimiza volatilidade
                            - **Max Return**: Maximiza retorno esperado
                            - **Risk Parity**: Equaliza contribuição de risco
                            - **Alpha Weighted**: Prioriza ativos com alfa positivo
                            """)
                            
                            backtest_btn = gr.Button(
                                "🧪 Executar Backtest Completo",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.TabItem("📋 Relatório Comparativo"):
                                    backtest_report_output = gr.Markdown(
                                        value="Configure os parâmetros e clique em 'Executar Backtest Completo'...",
                                        height=600
                                    )
                                
                                with gr.TabItem("📊 Performance Comparativa"):
                                    backtest_performance_plot = gr.Plot(
                                        label="Comparação de Performance"
                                    )
                                
                                with gr.TabItem("🎯 Métricas Multi-dimensionais"):
                                    backtest_metrics_plot = gr.Plot(
                                        label="Análise Radar de Estratégias"
                                    )

            
            # Conectar eventos V-Cov
            predict_btn.click(
                fn=self.predict_wrapper,
                inputs=[tickers_input, period_input, window_input],
                outputs=[vcov_plot, corr_plot, result_markdown],
                show_progress=True
            )
            
            # Conectar eventos Ponderação Alfa
            alpha_btn.click(
                fn=self.alpha_weighting_wrapper,
                inputs=[alpha_tickers_input, benchmark_input, risk_free_input],
                outputs=[weights_plot, alpha_beta_plot, alpha_result_markdown],
                show_progress=True
            )
            
            # Conectar eventos Modelo Híbrido
            hybrid_btn.click(
                fn=self.hybrid_portfolio_wrapper,
                inputs=[
                    hybrid_tickers_input,
                    hybrid_vcov_period,
                    hybrid_vcov_window,
                    hybrid_alpha_period,
                    hybrid_max_weight,
                    hybrid_min_weight,
                    hybrid_optimization_method,
                    hybrid_allow_short
                ],
                outputs=[hybrid_weights_plot, hybrid_metrics_plot, hybrid_report_output],
                show_progress=True
            )
            
            # Conectar eventos Backtest
            backtest_btn.click(
                fn=self.backtest_wrapper,
                inputs=[
                    backtest_tickers_input,
                    backtest_start_date,
                    backtest_end_date,
                    backtest_rebalance_freq,
                    backtest_initial_capital
                ],
                outputs=[backtest_performance_plot, backtest_metrics_plot, backtest_report_output],
                show_progress=True
            )
            

        
        return demo
    
    def launch(self, **kwargs):
        """Lança a interface Gradio."""
        demo = self.create_interface()
        return demo.launch(**kwargs)


# Função de conveniência para criar a interface
def create_gradio_interface():
    """Cria e retorna uma instância da interface Gradio."""
    interface = GradioInterface()
    return interface.create_interface()
