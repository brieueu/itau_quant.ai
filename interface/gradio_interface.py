"""
Interface Dashboard Simplificada para o Modelo Híbrido
Dashboard moderno focado exclusivamente no modelo híbrido LSTM + Alfa

Autor: Engenheiro de Software Sênior - Mercado Financeiro
"""

import gradio as gr
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from vcov_predictor import VCovPredictor
from hybrid_portfolio_model import HybridPortfolioModel, PortfolioConstraints


class GradioInterface:
    """Interface dashboard simplificada para o modelo híbrido."""
    
    def __init__(self):
        """Inicialização."""
        self.predictor = VCovPredictor()
        self.hybrid_model = HybridPortfolioModel()
    
    def hybrid_wrapper(self, tickers, vcov_period, vcov_window, alpha_period,
                      benchmark, max_weight, min_weight, optimization_method, 
                      allow_short, risk_free_rate):
        """Wrapper principal para o modelo híbrido."""
        try:
            # Configurar restrições
            constraints = PortfolioConstraints(
                max_weight=float(max_weight) / 100,
                min_weight=float(min_weight) / 100,
                allow_short=allow_short
            )
            
            # Atualizar configurações
            self.hybrid_model.benchmark = benchmark
            self.hybrid_model.risk_free_rate = float(risk_free_rate) / 100
            
            # Executar otimização
            results = self.hybrid_model.generate_portfolio_recommendation(
                tickers=tickers,
                vcov_period=int(vcov_period),
                vcov_window=int(vcov_window),
                alpha_period=alpha_period,
                constraints=constraints,
                optimization_method=optimization_method
            )
            
            # Gerar visualizações
            weights_fig = self._create_weights_chart(results)
            metrics_fig = self._create_metrics_chart(results)
            
            # Gerar relatório
            report = self._create_report(results)
            
            return weights_fig, metrics_fig, report
            
        except Exception as e:
            error_msg = f"❌ **Erro**: {str(e)}"
            empty_fig = go.Figure()
            return empty_fig, empty_fig, error_msg
    
    def _create_weights_chart(self, results):
        """Cria gráfico de pesos."""
        tickers = list(results.optimal_weights.keys())
        weights = [results.optimal_weights[ticker] * 100 for ticker in tickers]
        alphas = [results.alpha_contribution.get(ticker, 0) for ticker in tickers]
        
        colors = ['green' if alpha > 0 else 'red' if alpha < 0 else 'gray' for alpha in alphas]
        
        fig = go.Figure(data=go.Bar(
            x=tickers,
            y=weights,
            marker_color=colors,
            text=[f"{w:.1f}%" for w in weights],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="🎯 Alocação Otimizada do Portfólio",
            xaxis_title="Ativos",
            yaxis_title="Peso (%)",
            height=400
        )
        
        return fig
    
    def _create_metrics_chart(self, results):
        """Cria gráfico de métricas."""
        metrics = ['Sharpe', 'Retorno', 'Volatilidade', 'Diversificação']
        values = [
            min(results.sharpe_ratio * 20, 100),
            results.expected_return * 100,
            (1 - results.portfolio_volatility) * 100,
            results.diversification_ratio * 50
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Portfólio'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="📊 Métricas do Portfólio",
            height=400
        )
        
        return fig
    
    def _create_report(self, results):
        """Cria relatório."""
        total_alpha = sum(results.alpha_contribution.values())
        
        report = f"""
# 📊 Relatório de Otimização

## 🎯 Métricas Principais
- **Sharpe Ratio**: `{results.sharpe_ratio:.4f}`
- **Retorno Esperado**: `{results.expected_return*100:.2f}%` ao ano
- **Volatilidade**: `{results.portfolio_volatility*100:.2f}%` ao ano
- **Max Drawdown**: `{results.max_drawdown*100:.2f}%`

## 🎯 Alocação de Ativos
"""
        
        for ticker in sorted(results.optimal_weights.keys()):
            weight = results.optimal_weights[ticker] * 100
            alpha = results.alpha_contribution.get(ticker, 0) * 100
            report += f"- **{ticker}**: {weight:.1f}% (Alfa: {alpha:.3f}%)\n"
        
        report += f"""

## 📈 Análise
- **Alfa Total**: `{total_alpha*100:.3f}%`
- **Método**: {results.optimization_method.upper()}
- **Status**: {results.convergence_status}

*Relatório gerado pelo Sistema Híbrido LSTM + Alfa*
"""
        
        return report
    
    def create_interface(self):
        """Cria interface simplificada."""
        
        with gr.Blocks(title="🚀 Dashboard IA Quantitativa", theme=gr.themes.Soft()) as demo:
            
            gr.HTML("""
            <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 30px; border-radius: 15px; margin-bottom: 25px;">
                <h1>🚀 Dashboard de IA Quantitativa</h1>
                <h2>Sistema Híbrido: LSTM V-Cov + Otimização Alfa</h2>
            </div>
            """)
            
            with gr.Row():
                # Painel de controle
                with gr.Column(scale=1):
                    gr.Markdown("## 🎛️ Configurações")
                    
                    tickers_input = gr.Textbox(
                        label="Tickers",
                        value="AAPL, GOOGL, MSFT, AMZN, TSLA",
                        lines=2
                    )
                    
                    with gr.Row():
                        vcov_period = gr.Number(label="Período (anos)", value=3, minimum=1, maximum=10)
                        vcov_window = gr.Number(label="Janela (dias)", value=60, minimum=30, maximum=252)
                    
                    with gr.Row():
                        alpha_period = gr.Dropdown(
                            label="Período Alfa", 
                            choices=["1y", "2y", "3y", "5y"], 
                            value="3y"
                        )
                        benchmark = gr.Dropdown(
                            label="Benchmark",
                            choices=["^GSPC", "^IXIC", "^DJI"],
                            value="^GSPC"
                        )
                    
                    with gr.Row():
                        max_weight = gr.Slider(label="Peso Máx (%)", minimum=15, maximum=50, value=25, step=5)
                        min_weight = gr.Slider(label="Peso Min (%)", minimum=0, maximum=20, value=3, step=1)
                    
                    optimization_method = gr.Radio(
                        label="Estratégia",
                        choices=[
                            ("Máximo Sharpe", "max_sharpe"),
                            ("Mínima Variância", "min_variance"),
                            ("Máximo Retorno", "max_return"),
                            ("Risk Parity", "risk_parity"),
                            ("Alpha Weighted", "alpha_weighted")
                        ],
                        value="max_sharpe"
                    )
                    
                    with gr.Accordion("Avançado", open=False):
                        allow_short = gr.Checkbox(label="Permitir Short", value=False)
                        risk_free_rate = gr.Number(label="Taxa Livre Risco (%)", value=4.5)
                    
                    execute_btn = gr.Button("EXECUTAR OTIMIZAÇÃO", variant="primary", size="lg")
                
                # Resultados
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("🎯 Alocação"):
                            weights_plot = gr.Plot()
                        
                        with gr.TabItem("📊 Métricas"):
                            metrics_plot = gr.Plot()
                        
                        with gr.TabItem("📋 Relatório"):
                            report_output = gr.Markdown(
                                value="Configure os parâmetros e execute a otimização."
                            )
            
            # Conectar eventos
            execute_btn.click(
                fn=self.hybrid_wrapper,
                inputs=[
                    tickers_input, vcov_period, vcov_window, alpha_period,
                    benchmark, max_weight, min_weight, optimization_method,
                    allow_short, risk_free_rate
                ],
                outputs=[weights_plot, metrics_plot, report_output],
                show_progress=True
            )
        
        return demo
    
    def launch(self, **kwargs):
        """Lança a interface."""
        demo = self.create_interface()
        return demo.launch(**kwargs)
