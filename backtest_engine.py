"""
Sistema de Backtest Completo para Estratégias de Otimização
Testa todas as opções do modelo híbrido com métricas detalhadas de performance.

Autor: Engenheiro Quantitativo Senior
Especialização: Backtesting, Análise de Performance, Risk Management
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from hybrid_portfolio_model import HybridPortfolioModel
from vcov_predictor import VCovPredictor
from alpha_weighting import AlphaWeighting

@dataclass
class BacktestConfig:
    """Configuração do backtest."""
    start_date: str
    end_date: str
    rebalance_frequency: int = 30  # dias
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% por transação
    benchmark: str = "^GSPC"
    risk_free_rate: float = 0.02

@dataclass
class PerformanceMetrics:
    """Métricas de performance do backtest."""
    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float
    cvar_95: float
    final_value: float
    benchmark_return: float


class BacktestEngine:
    """Engine completo de backtest para todas as estratégias."""
    
    def __init__(self, config: BacktestConfig):
        """
        Inicializa o engine de backtest.
        
        Args:
            config: Configuração do backtest
        """
        self.config = config
        self.hybrid_model = HybridPortfolioModel()
        self.results = {}
        
        # Estratégias a serem testadas
        self.strategies = [
            'max_sharpe',
            'min_variance', 
            'max_return',
            'risk_parity',
            'alpha_weighted'
        ]
    
    def run_full_backtest(self, tickers: List[str], 
                         vcov_window: int = 90,
                         alpha_period: str = "2y") -> Dict[str, PerformanceMetrics]:
        """
        Executa backtest completo para todas as estratégias.
        
        Args:
            tickers: Lista de tickers para análise
            vcov_window: Janela para predição V-Cov
            alpha_period: Período para cálculo do alfa
            
        Returns:
            Dicionário com métricas de todas as estratégias
        """
        print("🚀 Iniciando backtest completo...")
        print(f"📅 Período: {self.config.start_date} - {self.config.end_date}")
        print(f"📈 Ativos: {', '.join(tickers)}")
        print(f"🔄 Rebalanceamento: {self.config.rebalance_frequency} dias")
        print(f"💰 Capital inicial: ${self.config.initial_capital:,.2f}")
        print()
        
        # Baixar dados históricos
        price_data = self._download_historical_data(tickers)
        benchmark_data = self._download_benchmark_data()
        
        # Datas de rebalanceamento
        rebalance_dates = self._generate_rebalance_dates()
        
        results = {}
        
        # Testar cada estratégia
        for strategy in self.strategies:
            print(f"🧪 Testando estratégia: {strategy.upper()}")
            
            portfolio_values, weights_history = self._run_strategy_backtest(
                strategy, tickers, price_data, rebalance_dates, 
                vcov_window, alpha_period
            )
            
            if portfolio_values is not None:
                metrics = self._calculate_performance_metrics(
                    strategy, portfolio_values, benchmark_data, weights_history
                )
                results[strategy] = metrics
                
                print(f"✅ {strategy}: Retorno = {metrics.total_return:.2%}, Sharpe = {metrics.sharpe_ratio:.2f}")
            else:
                print(f"❌ Falha na estratégia: {strategy}")
            
            print()
        
        # Adicionar benchmark
        benchmark_metrics = self._calculate_benchmark_metrics(benchmark_data)
        results['benchmark'] = benchmark_metrics
        
        print("📊 Backtest completo concluído!")
        return results
    
    def _download_historical_data(self, tickers: List[str]) -> pd.DataFrame:
        """Baixa dados históricos dos ativos."""
        print("📊 Baixando dados históricos dos ativos...")
        
        try:
            data = yf.download(
                tickers,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                auto_adjust=True
            )
            
            if len(tickers) == 1:
                prices = data['Close'].to_frame()
                prices.columns = tickers
            else:
                prices = data['Close']
            
            return prices.dropna()
            
        except Exception as e:
            raise ValueError(f"Erro ao baixar dados: {str(e)}")
    
    def _download_benchmark_data(self) -> pd.Series:
        """Baixa dados do benchmark."""
        print(f"📊 Baixando dados do benchmark ({self.config.benchmark})...")
        
        try:
            data = yf.download(
                self.config.benchmark,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                auto_adjust=True
            )
            
            return data['Close'].dropna()
            
        except Exception as e:
            raise ValueError(f"Erro ao baixar dados do benchmark: {str(e)}")
    
    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        """Gera datas de rebalanceamento."""
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        
        dates = []
        current = start
        
        while current <= end:
            dates.append(current)
            current += timedelta(days=self.config.rebalance_frequency)
        
        return dates
    
    def _run_strategy_backtest(self, strategy: str, tickers: List[str], 
                             price_data: pd.DataFrame, rebalance_dates: List[pd.Timestamp],
                             vcov_window: int, alpha_period: str) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Executa backtest de uma estratégia específica.
        
        Args:
            strategy: Nome da estratégia
            tickers: Lista de tickers
            price_data: Dados de preços
            rebalance_dates: Datas de rebalanceamento
            vcov_window: Janela V-Cov
            alpha_period: Período alfa
            
        Returns:
            Tupla (valores do portfólio, histórico de pesos)
        """
        try:
            portfolio_value = self.config.initial_capital
            portfolio_values = []
            weights_history = []
            current_weights = None
            
            for i, date in enumerate(rebalance_dates):
                # Encontrar data mais próxima nos dados
                available_dates = price_data.index[price_data.index <= date]
                if len(available_dates) == 0:
                    continue
                
                current_date = available_dates[-1]
                
                # Obter preços atuais
                current_prices = price_data.loc[current_date]
                
                # Calcular pesos para o período
                if i == 0 or current_weights is None:
                    # Primeira alocação - usar pesos iguais inicialmente
                    new_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
                else:
                    # Gerar novos pesos usando o modelo híbrido
                    new_weights = self._get_strategy_weights(
                        strategy, tickers, current_date, price_data,
                        vcov_window, alpha_period
                    )
                
                # Calcular custos de transação se houver mudança de pesos
                if current_weights is not None:
                    transaction_cost = self._calculate_transaction_costs(
                        current_weights, new_weights, portfolio_value
                    )
                    portfolio_value -= transaction_cost
                
                current_weights = new_weights
                weights_history.append({
                    'date': current_date,
                    **current_weights
                })
                
                # Calcular performance até próximo rebalanceamento
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    available_next = price_data.index[
                        (price_data.index > current_date) & (price_data.index <= next_date)
                    ]
                else:
                    available_next = price_data.index[price_data.index > current_date]
                
                # Simular performance do portfólio
                for next_date in available_next:
                    if next_date in price_data.index:
                        next_prices = price_data.loc[next_date]
                        
                        # Calcular retorno do portfólio
                        returns = (next_prices / current_prices - 1)
                        portfolio_return = sum(
                            current_weights.get(ticker, 0) * returns[ticker] 
                            for ticker in tickers if ticker in returns.index
                        )
                        
                        portfolio_value *= (1 + portfolio_return)
                        portfolio_values.append({
                            'date': next_date,
                            'value': portfolio_value
                        })
                        
                        current_prices = next_prices
            
            # Converter para pandas
            if portfolio_values:
                portfolio_df = pd.DataFrame(portfolio_values)
                portfolio_series = pd.Series(
                    data=portfolio_df['value'].values,
                    index=portfolio_df['date'],
                    name=strategy
                )
                
                weights_df = pd.DataFrame(weights_history)
                weights_df.set_index('date', inplace=True)
                
                return portfolio_series, weights_df
            else:
                return None, None
                
        except Exception as e:
            print(f"❌ Erro na estratégia {strategy}: {str(e)}")
            return None, None
    
    def _get_strategy_weights(self, strategy: str, tickers: List[str], 
                            current_date: pd.Timestamp, price_data: pd.DataFrame,
                            vcov_window: int, alpha_period: str) -> Dict[str, float]:
        """
        Obtém pesos da estratégia usando o modelo híbrido.
        
        Args:
            strategy: Nome da estratégia
            tickers: Lista de tickers
            current_date: Data atual
            price_data: Dados de preços
            vcov_window: Janela V-Cov
            alpha_period: Período alfa
            
        Returns:
            Dicionário com pesos dos ativos
        """
        try:
            # Obter dados históricos até a data atual
            historical_data = price_data.loc[price_data.index <= current_date]
            
            if len(historical_data) < vcov_window + 30:  # Mínimo de dados necessários
                # Retornar pesos iguais se não houver dados suficientes
                return {ticker: 1.0/len(tickers) for ticker in tickers}
            
            # Usar modelo híbrido para gerar recomendação
            tickers_str = ','.join(tickers)
            
            result = self.hybrid_model.generate_portfolio_recommendation(
                tickers=tickers_str,
                optimization_method=strategy,
                vcov_window=vcov_window,
                alpha_period=alpha_period
            )
            
            # O resultado é um objeto OptimizationResults, não um dict
            if result and hasattr(result, 'optimal_weights'):
                weights = result.optimal_weights
                
                # Normalizar pesos para somar 1
                total_weight = sum(weights.values())
                if total_weight > 0:
                    normalized_weights = {
                        ticker: weight / total_weight 
                        for ticker, weight in weights.items()
                    }
                    return normalized_weights
            
            # Fallback para pesos iguais
            return {ticker: 1.0/len(tickers) for ticker in tickers}
            
        except Exception as e:
            print(f"⚠️ Erro ao obter pesos para {strategy}: {str(e)}")
            return {ticker: 1.0/len(tickers) for ticker in tickers}
    
    def _calculate_transaction_costs(self, old_weights: Dict[str, float], 
                                   new_weights: Dict[str, float], 
                                   portfolio_value: float) -> float:
        """Calcula custos de transação."""
        total_turnover = 0.0
        
        for ticker in set(list(old_weights.keys()) + list(new_weights.keys())):
            old_weight = old_weights.get(ticker, 0.0)
            new_weight = new_weights.get(ticker, 0.0)
            total_turnover += abs(new_weight - old_weight)
        
        return total_turnover * portfolio_value * self.config.transaction_cost
    
    def _calculate_performance_metrics(self, strategy_name: str, 
                                     portfolio_values: pd.Series,
                                     benchmark_data: pd.Series,
                                     weights_history: pd.DataFrame) -> PerformanceMetrics:
        """Calcula métricas de performance detalhadas."""
        
        # Alinhar dados
        common_dates = portfolio_values.index.intersection(benchmark_data.index)
        portfolio_aligned = portfolio_values.reindex(common_dates)
        benchmark_aligned = benchmark_data.reindex(common_dates)
        
        # Calcular retornos
        portfolio_returns = portfolio_aligned.pct_change().dropna()
        benchmark_returns = benchmark_aligned.pct_change().dropna()
        
        # Métricas básicas
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        trading_days = len(portfolio_returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = portfolio_returns - self.config.risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Alpha e Beta vs benchmark
        if len(benchmark_returns) > 1 and len(portfolio_returns) > 1:
            # Alinhar retornos - garantir que sejam Series 1D
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_aligned = portfolio_returns.reindex(common_dates).dropna()
            benchmark_aligned = benchmark_returns.reindex(common_dates).dropna()
            
            # Garantir que ambas sejam Series 1D
            if hasattr(portfolio_aligned, 'values'):
                portfolio_vals = portfolio_aligned.values.flatten()
            else:
                portfolio_vals = portfolio_aligned
                
            if hasattr(benchmark_aligned, 'values'):
                benchmark_vals = benchmark_aligned.values.flatten()
            else:
                benchmark_vals = benchmark_aligned
            
            # Alinhar tamanhos
            min_len = min(len(portfolio_vals), len(benchmark_vals))
            portfolio_vals = portfolio_vals[:min_len]
            benchmark_vals = benchmark_vals[:min_len]
            
            # Criar DataFrame alinhado
            aligned_returns = pd.DataFrame({
                'portfolio': portfolio_vals,
                'benchmark': benchmark_vals
            }).dropna()
            
            if len(aligned_returns) > 1:
                covariance = np.cov(aligned_returns['portfolio'], aligned_returns['benchmark'])[0,1]
                benchmark_variance = np.var(aligned_returns['benchmark'])
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                portfolio_mean = aligned_returns['portfolio'].mean() * 252
                benchmark_mean = aligned_returns['benchmark'].mean() * 252
                alpha = portfolio_mean - (self.config.risk_free_rate + beta * (benchmark_mean - self.config.risk_free_rate))
                
                # Information ratio
                active_returns = aligned_returns['portfolio'] - aligned_returns['benchmark']
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            else:
                alpha = beta = information_ratio = 0
        else:
            alpha = beta = information_ratio = 0
        
        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        # Profit factor
        positive_returns = portfolio_returns[portfolio_returns > 0].sum()
        negative_returns = abs(portfolio_returns[portfolio_returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else np.inf
        
        # VaR e CVaR (95%)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Benchmark return
        benchmark_total_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) - 1
        
        return PerformanceMetrics(
            strategy_name=strategy_name,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            final_value=portfolio_values.iloc[-1],
            benchmark_return=benchmark_total_return
        )
    
    def _calculate_benchmark_metrics(self, benchmark_data: pd.Series) -> PerformanceMetrics:
        """Calcula métricas do benchmark."""
        returns = benchmark_data.pct_change().dropna()
        
        total_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[0]) - 1
        trading_days = len(returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        volatility = returns.std() * np.sqrt(252)
        
        excess_returns = returns - self.config.risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        win_rate = (returns > 0).sum() / len(returns)
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else np.inf
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        final_value = self.config.initial_capital * (1 + total_return)
        
        return PerformanceMetrics(
            strategy_name="benchmark",
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            alpha=0.0,  # Benchmark não tem alfa contra si mesmo
            beta=1.0,   # Benchmark tem beta 1 contra si mesmo
            information_ratio=0.0,
            win_rate=win_rate,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            final_value=final_value,
            benchmark_return=total_return
        )
    
    def generate_comparison_report(self, results: Dict[str, PerformanceMetrics]) -> str:
        """Gera relatório comparativo das estratégias."""
        report = "# 📊 RELATÓRIO DE BACKTEST COMPARATIVO\n\n"
        
        report += f"**Período:** {self.config.start_date} - {self.config.end_date}\n"
        report += f"**Capital Inicial:** ${self.config.initial_capital:,.2f}\n"
        report += f"**Rebalanceamento:** {self.config.rebalance_frequency} dias\n"
        report += f"**Custos de Transação:** {self.config.transaction_cost:.2%}\n\n"
        
        # Tabela resumo
        report += "## 📈 Resumo de Performance\n\n"
        report += "| Estratégia | Retorno Total | Retorno Anual | Volatilidade | Sharpe | Max DD | Calmar |\n"
        report += "|------------|---------------|---------------|--------------|---------|---------|--------|\n"
        
        # Ordenar por Sharpe ratio
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].sharpe_ratio,
            reverse=True
        )
        
        for strategy_name, metrics in sorted_results:
            emoji = "🏆" if strategy_name != "benchmark" and metrics.sharpe_ratio == max(r[1].sharpe_ratio for r in sorted_results if r[0] != "benchmark") else ""
            
            report += f"| **{strategy_name.upper()}** {emoji} | {metrics.total_return:.2%} | {metrics.annualized_return:.2%} | {metrics.volatility:.2%} | {metrics.sharpe_ratio:.2f} | {metrics.max_drawdown:.2%} | {metrics.calmar_ratio:.2f} |\n"
        
        # Métricas detalhadas
        report += "\n## 🔍 Métricas Detalhadas\n\n"
        
        for strategy_name, metrics in sorted_results:
            if strategy_name == "benchmark":
                continue
                
            report += f"### 📊 {strategy_name.upper()}\n\n"
            report += f"- **Valor Final:** ${metrics.final_value:,.2f}\n"
            report += f"- **Alfa:** {metrics.alpha:.2%}\n"
            report += f"- **Beta:** {metrics.beta:.2f}\n"
            report += f"- **Information Ratio:** {metrics.information_ratio:.2f}\n"
            report += f"- **Sortino Ratio:** {metrics.sortino_ratio:.2f}\n"
            report += f"- **Taxa de Acerto:** {metrics.win_rate:.2%}\n"
            report += f"- **Profit Factor:** {metrics.profit_factor:.2f}\n"
            report += f"- **VaR (95%):** {metrics.var_95:.2%}\n"
            report += f"- **CVaR (95%):** {metrics.cvar_95:.2%}\n\n"
        
        # Análise comparativa
        report += "## 🎯 Análise Comparativa\n\n"
        
        # Melhor estratégia por métrica
        best_return = max(results.items(), key=lambda x: x[1].total_return if x[0] != "benchmark" else -np.inf)
        best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio if x[0] != "benchmark" else -np.inf)
        best_calmar = max(results.items(), key=lambda x: x[1].calmar_ratio if x[0] != "benchmark" else -np.inf)
        lowest_dd = min(results.items(), key=lambda x: abs(x[1].max_drawdown) if x[0] != "benchmark" else np.inf)
        
        report += f"🏆 **Maior Retorno:** {best_return[0].upper()} ({best_return[1].total_return:.2%})\n\n"
        report += f"🎯 **Melhor Sharpe:** {best_sharpe[0].upper()} ({best_sharpe[1].sharpe_ratio:.2f})\n\n"
        report += f"📈 **Melhor Calmar:** {best_calmar[0].upper()} ({best_calmar[1].calmar_ratio:.2f})\n\n"
        report += f"🛡️ **Menor Drawdown:** {lowest_dd[0].upper()} ({lowest_dd[1].max_drawdown:.2%})\n\n"
        
        # Benchmark comparison
        if "benchmark" in results:
            benchmark = results["benchmark"]
            report += "## 📊 vs. Benchmark\n\n"
            
            strategies_beating_benchmark = [
                (name, metrics) for name, metrics in results.items()
                if name != "benchmark" and metrics.total_return > benchmark.total_return
            ]
            
            report += f"**Estratégias que superaram o benchmark:** {len(strategies_beating_benchmark)}/{len(results)-1}\n\n"
            
            for name, metrics in strategies_beating_benchmark:
                excess_return = metrics.total_return - benchmark.total_return
                report += f"- **{name.upper()}:** +{excess_return:.2%} vs benchmark\n"
        
        report += "\n## 💡 Recomendações\n\n"
        
        if best_sharpe[1].sharpe_ratio > 1.0:
            report += f"✅ A estratégia **{best_sharpe[0].upper()}** apresenta excelente relação risco-retorno (Sharpe > 1.0)\n\n"
        
        if abs(lowest_dd[1].max_drawdown) < 0.10:
            report += f"🛡️ A estratégia **{lowest_dd[0].upper()}** oferece boa proteção contra perdas (DD < 10%)\n\n"
        
        report += "- Considere diversificar entre as melhores estratégias\n"
        report += "- Monitore regularmente as métricas de risco\n"
        report += "- Ajuste a frequência de rebalanceamento conforme volatilidade do mercado\n"
        
        return report

def run_comprehensive_backtest(tickers: List[str], 
                             start_date: str = "2020-01-01",
                             end_date: str = "2024-01-01",
                             rebalance_frequency: int = 30,
                             initial_capital: float = 100000.0) -> Dict[str, PerformanceMetrics]:
    """
    Função principal para executar backtest completo.
    
    Args:
        tickers: Lista de tickers para análise
        start_date: Data inicial do backtest
        end_date: Data final do backtest
        rebalance_frequency: Frequência de rebalanceamento em dias
        initial_capital: Capital inicial
        
    Returns:
        Dicionário com resultados de todas as estratégias
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=rebalance_frequency,
        initial_capital=initial_capital
    )
    
    engine = BacktestEngine(config)
    return engine.run_full_backtest(tickers)
