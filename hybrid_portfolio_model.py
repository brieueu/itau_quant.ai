"""
Modelo Híbrido: Previsão V-Cov + Ponderação Alfa
Sistema integrado que combina previsão de matrizes de variância-covariância
com análise de alfa para otimização de portfólio quantitativo.

Autor: Engenheiro Quantitativo Senior
Especialização: LSTM, Otimização de Portfólio, Análise Quantitativa
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import LinAlgError
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from vcov_predictor import VCovPredictor
from alpha_weighting import AlphaWeighting

warnings.filterwarnings('ignore')


@dataclass
class PortfolioConstraints:
    """Classe para definir restrições do portfólio."""
    max_weight: float = 0.4  # Peso máximo por ativo
    min_weight: float = 0.0  # Peso mínimo por ativo
    max_concentration: float = 0.6  # Concentração máxima total
    target_volatility: Optional[float] = None  # Volatilidade alvo
    min_expected_return: Optional[float] = None  # Retorno mínimo esperado
    allow_short: bool = False  # Permitir posições vendidas


@dataclass
class OptimizationResults:
    """Classe para armazenar resultados da otimização."""
    optimal_weights: Dict[str, float]
    expected_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    alpha_contribution: Dict[str, float]
    risk_contribution: Dict[str, float]
    diversification_ratio: float
    optimization_method: str
    convergence_status: str


class HybridPortfolioModel:
    """
    Modelo híbrido que combina previsão LSTM de matrizes V-Cov 
    com análise alfa para otimização quantitativa de portfólio.
    """
    
    def __init__(self, benchmark: str = "^GSPC", risk_free_rate: float = 0.02):
        """
        Inicializa o modelo híbrido.
        
        Args:
            benchmark: Índice de referência para cálculo do alfa
            risk_free_rate: Taxa livre de risco anualizada
        """
        self.vcov_predictor = VCovPredictor()
        self.alpha_calculator = AlphaWeighting(benchmark=benchmark, risk_free_rate=risk_free_rate)
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        
        # Resultados armazenados
        self.predicted_vcov = None
        self.alpha_data = None
        self.expected_returns = None
        self.portfolio_results = None
        
        print(f"🚀 Modelo Híbrido inicializado - Benchmark: {benchmark}, RF: {risk_free_rate:.2%}")

    def generate_portfolio_recommendation(
        self, 
        tickers: Union[str, List[str]], 
        vcov_period: int = 2, 
        vcov_window: int = 60,
        alpha_period: str = "2y",
        constraints: Optional[PortfolioConstraints] = None,
        optimization_method: str = "max_sharpe"
    ) -> OptimizationResults:
        """
        Gera recomendação completa de portfólio usando o modelo híbrido.
        
        Args:
            tickers: Lista de ativos ou string separada por vírgula
            vcov_period: Período em anos para previsão V-Cov
            vcov_window: Janela em dias para V-Cov
            alpha_period: Período para análise alfa
            constraints: Restrições do portfólio
            optimization_method: Método de otimização
            
        Returns:
            OptimizationResults: Resultados completos da otimização
        """
        # Processar tickers
        if isinstance(tickers, str):
            ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        else:
            ticker_list = [t.upper() for t in tickers]
        
        print(f"📊 Iniciando análise híbrida para {len(ticker_list)} ativos...")
        
        # Etapa 1: Previsão da Matriz V-Cov usando LSTM
        print("🔮 Gerando previsão V-Cov com LSTM...")
        vcov_result = self.vcov_predictor.predict_vcov_matrix(
            ','.join(ticker_list), 
            vcov_period, 
            vcov_window
        )
        
        if not vcov_result['success']:
            raise ValueError(f"Erro na previsão V-Cov: {vcov_result['error']}")
        
        self.predicted_vcov = vcov_result['vcov_matrix']
        
        # Etapa 2: Análise de Alfa
        print("📈 Calculando alfas dos ativos...")
        alpha_weights = self.alpha_calculator.calculate_portfolio_weights(ticker_list, alpha_period)
        self.alpha_data = {
            'weights': alpha_weights,
            'alphas': self.alpha_calculator.alphas.copy(),
            'betas': self.alpha_calculator.betas.copy()
        }
        
        # Etapa 3: Estimativa de retornos esperados
        print("💰 Estimando retornos esperados...")
        self.expected_returns = self._estimate_expected_returns(ticker_list)
        
        # Etapa 4: Otimização do portfólio
        print(f"⚡ Otimizando portfólio ({optimization_method})...")
        if constraints is None:
            constraints = PortfolioConstraints()
            
        optimal_weights = self._optimize_portfolio(
            ticker_list, 
            constraints, 
            optimization_method
        )
        
        # Etapa 5: Calcular métricas do portfólio
        print("📊 Calculando métricas de performance...")
        portfolio_metrics = self._calculate_portfolio_metrics(ticker_list, optimal_weights)
        
        # Compilar resultados
        results = OptimizationResults(
            optimal_weights=optimal_weights,
            expected_return=portfolio_metrics['expected_return'],
            portfolio_volatility=portfolio_metrics['volatility'],
            sharpe_ratio=portfolio_metrics['sharpe_ratio'],
            max_drawdown=portfolio_metrics['max_drawdown'],
            var_95=portfolio_metrics['var_95'],
            cvar_95=portfolio_metrics['cvar_95'],
            alpha_contribution=portfolio_metrics['alpha_contribution'],
            risk_contribution=portfolio_metrics['risk_contribution'],
            diversification_ratio=portfolio_metrics['diversification_ratio'],
            optimization_method=optimization_method,
            convergence_status="Success"
        )
        
        self.portfolio_results = results
        print("✅ Análise híbrida concluída!")
        
        return results
    
    def _estimate_expected_returns(self, tickers: List[str]) -> Dict[str, float]:
        """
        Estima retornos esperados combinando CAPM e informações de alfa.
        
        Args:
            tickers: Lista de tickers
            
        Returns:
            Dict com retornos esperados por ativo
        """
        expected_returns = {}
        
        # Retorno esperado do mercado (assumindo prêmio de risco de 6%)
        market_return = self.risk_free_rate + 0.06
        
        for ticker in tickers:
            # CAPM: E(R) = RF + Beta * (RM - RF)
            beta = self.alpha_data['betas'].get(ticker, 1.0)
            alpha = self.alpha_data['alphas'].get(ticker, 0.0)
            
            # Combinar CAPM com alfa histórico
            capm_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
            expected_return = capm_return + alpha
            
            expected_returns[ticker] = expected_return
        
        return expected_returns
    
    def _optimize_portfolio(
        self, 
        tickers: List[str], 
        constraints: PortfolioConstraints,
        method: str
    ) -> Dict[str, float]:
        """
        Otimiza o portfólio usando diferentes métodos.
        
        Args:
            tickers: Lista de tickers
            constraints: Restrições do portfólio
            method: Método de otimização
            
        Returns:
            Dict com pesos otimizados
        """
        n_assets = len(tickers)
        
        # Função objetivo baseada no método escolhido
        if method == "max_sharpe":
            objective = self._negative_sharpe_ratio
        elif method == "min_variance":
            objective = self._portfolio_variance
        elif method == "max_return":
            objective = self._negative_portfolio_return
        elif method == "risk_parity":
            objective = self._risk_parity_objective
        elif method == "alpha_weighted":
            objective = self._alpha_weighted_objective
        else:
            objective = self._negative_sharpe_ratio
        
        # Restrições de otimização
        scipy_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Soma = 1
        ]
        
        # Limites de peso
        if constraints.allow_short:
            bounds = [(-constraints.max_weight, constraints.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Restrição de concentração
        if constraints.max_concentration < 1.0:
            scipy_constraints.append({
                'type': 'ineq', 
                'fun': lambda w: constraints.max_concentration - np.max(np.abs(w))
            })
        
        # Restrição de volatilidade alvo
        if constraints.target_volatility is not None:
            scipy_constraints.append({
                'type': 'eq',
                'fun': lambda w: self._portfolio_volatility(w) - constraints.target_volatility
            })
        
        # Restrição de retorno mínimo
        if constraints.min_expected_return is not None:
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w: self._portfolio_return(w) - constraints.min_expected_return
            })
        
        # Chute inicial (equally weighted)
        x0 = np.ones(n_assets) / n_assets
        
        # Otimização
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=scipy_constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                print(f"⚠️ Otimização não convergiu, usando pesos iguais")
                optimal_weights = x0
                
        except Exception as e:
            print(f"❌ Erro na otimização: {e}")
            optimal_weights = x0
        
        # Normalizar pesos
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        return dict(zip(tickers, optimal_weights))
    
    def _negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calcula o negativo do Sharpe ratio (para maximização)."""
        portfolio_return = self._portfolio_return(weights)
        portfolio_vol = self._portfolio_volatility(weights)
        
        if portfolio_vol == 0:
            return -np.inf
        
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        return -sharpe
    
    def _portfolio_variance(self, weights: np.ndarray) -> float:
        """Calcula a variância do portfólio."""
        return np.dot(weights, np.dot(self.predicted_vcov, weights))
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calcula a volatilidade do portfólio."""
        return np.sqrt(self._portfolio_variance(weights))
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calcula o retorno esperado do portfólio."""
        returns_array = np.array(list(self.expected_returns.values()))
        return np.dot(weights, returns_array)
    
    def _negative_portfolio_return(self, weights: np.ndarray) -> float:
        """Negativo do retorno do portfólio (para maximização)."""
        return -self._portfolio_return(weights)
    
    def _risk_parity_objective(self, weights: np.ndarray) -> float:
        """Objetivo para Risk Parity (equalizar contribuições de risco)."""
        portfolio_vol = self._portfolio_volatility(weights)
        if portfolio_vol == 0:
            return np.inf
        
        # Contribuições de risco
        marginal_contrib = np.dot(self.predicted_vcov, weights) / portfolio_vol
        contrib = weights * marginal_contrib
        
        # Minimizar a variação das contribuições
        target_contrib = portfolio_vol / len(weights)
        return np.sum((contrib - target_contrib) ** 2)
    
    def _alpha_weighted_objective(self, weights: np.ndarray) -> float:
        """Objetivo que penaliza desvios dos pesos baseados em alfa."""
        alpha_weights = np.array([self.alpha_data['weights'].get(ticker, 1/len(weights)) 
                                 for ticker in self.expected_returns.keys()])
        
        # Penalizar desvios dos pesos alfa + penalizar risco
        alpha_penalty = np.sum((weights - alpha_weights) ** 2)
        risk_penalty = self._portfolio_variance(weights) * 0.5
        
        return alpha_penalty + risk_penalty
    
    def _calculate_portfolio_metrics(
        self, 
        tickers: List[str], 
        weights: Dict[str, float]
    ) -> Dict:
        """
        Calcula métricas completas do portfólio otimizado.
        
        Args:
            tickers: Lista de tickers
            weights: Pesos do portfólio
            
        Returns:
            Dict com métricas do portfólio
        """
        w = np.array([weights[ticker] for ticker in tickers])
        
        # Métricas básicas
        expected_return = self._portfolio_return(w)
        volatility = self._portfolio_volatility(w)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # VaR e CVaR (assumindo distribuição normal)
        var_95 = -expected_return + 1.645 * volatility  # 95% VaR
        cvar_95 = var_95 + volatility * 0.399  # Aproximação para CVaR
        
        # Contribuições de alfa
        alpha_contribution = {}
        for i, ticker in enumerate(tickers):
            alpha_contrib = weights[ticker] * self.alpha_data['alphas'].get(ticker, 0)
            alpha_contribution[ticker] = alpha_contrib
        
        # Contribuições de risco
        risk_contribution = {}
        marginal_contrib = np.dot(self.predicted_vcov, w)
        for i, ticker in enumerate(tickers):
            risk_contrib = weights[ticker] * marginal_contrib[i] / (volatility ** 2)
            risk_contribution[ticker] = risk_contrib
        
        # Ratio de diversificação
        individual_vols = np.sqrt(np.diag(self.predicted_vcov))
        weighted_avg_vol = np.sum(w * individual_vols)
        diversification_ratio = weighted_avg_vol / volatility if volatility > 0 else 1
        
        # Max Drawdown (estimativa baseada em volatilidade)
        max_drawdown = volatility * np.sqrt(2 / np.pi) * 2  # Aproximação
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'alpha_contribution': alpha_contribution,
            'risk_contribution': risk_contribution,
            'diversification_ratio': diversification_ratio
        }
    
    def generate_detailed_report(self) -> str:
        """
        Gera relatório detalhado da análise híbrida.
        
        Returns:
            String formatada com relatório completo
        """
        if self.portfolio_results is None:
            return "❌ Execute primeiro a análise híbrida com generate_portfolio_recommendation()"
        
        results = self.portfolio_results
        
        report = "# 🚀 RELATÓRIO HÍBRIDO: PREVISÃO V-COV + ALFA\n\n"
        
        # Informações gerais
        report += f"**Data da análise:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Método de otimização:** {results.optimization_method}\n"
        report += f"**Status:** {results.convergence_status}\n"
        report += f"**Benchmark:** {self.benchmark}\n"
        report += f"**Taxa livre de risco:** {self.risk_free_rate:.2%}\n\n"
        
        # Métricas do portfólio
        report += "## 📊 MÉTRICAS DO PORTFÓLIO OTIMIZADO\n\n"
        report += f"- **Retorno Esperado:** {results.expected_return:.2%} ao ano\n"
        report += f"- **Volatilidade:** {results.portfolio_volatility:.2%} ao ano\n"
        report += f"- **Sharpe Ratio:** {results.sharpe_ratio:.3f}\n"
        report += f"- **Value at Risk (95%):** {results.var_95:.2%}\n"
        report += f"- **CVaR (95%):** {results.cvar_95:.2%}\n"
        report += f"- **Max Drawdown Estimado:** {results.max_drawdown:.2%}\n"
        report += f"- **Ratio de Diversificação:** {results.diversification_ratio:.2f}\n\n"
        
        # Alocação otimizada
        report += "## 🎯 ALOCAÇÃO OTIMIZADA\n\n"
        report += "| Ativo | Peso | Alpha | Contrib. Alpha | Contrib. Risco |\n"
        report += "|-------|------|-------|----------------|----------------|\n"
        
        for ticker, weight in results.optimal_weights.items():
            alpha = self.alpha_data['alphas'].get(ticker, 0)
            alpha_contrib = results.alpha_contribution.get(ticker, 0)
            risk_contrib = results.risk_contribution.get(ticker, 0)
            
            report += f"| **{ticker}** | {weight:.2%} | {alpha:.4f} | {alpha_contrib:.4f} | {risk_contrib:.2%} |\n"
        
        # Análise de risco
        report += "\n## ⚠️ ANÁLISE DE RISCO\n\n"
        
        # Identificar maiores concentrações
        max_weight_ticker = max(results.optimal_weights.keys(), 
                               key=lambda x: results.optimal_weights[x])
        max_weight = results.optimal_weights[max_weight_ticker]
        
        report += f"- **Maior concentração:** {max_weight_ticker} ({max_weight:.2%})\n"
        
        # Análise de contribuições
        total_alpha_contrib = sum(results.alpha_contribution.values())
        report += f"- **Contribuição total de alfa:** {total_alpha_contrib:.4f}\n"
        
        # Ativos com alfa positivo
        positive_alpha_assets = [ticker for ticker, alpha in self.alpha_data['alphas'].items() 
                                if alpha > 0]
        report += f"- **Ativos com alfa positivo:** {len(positive_alpha_assets)} de {len(results.optimal_weights)}\n"
        
        # Recomendações
        report += "\n## 💡 RECOMENDAÇÕES\n\n"
        
        if results.sharpe_ratio > 1.0:
            report += "✅ **Portfólio atrativo**: Sharpe ratio superior a 1.0\n"
        else:
            report += "⚠️ **Portfólio moderado**: Considere revisar expectativas de retorno\n"
        
        if results.diversification_ratio > 1.2:
            report += "✅ **Boa diversificação**: Benefício de diversificação significativo\n"
        else:
            report += "⚠️ **Diversificação limitada**: Considere incluir mais ativos\n"
        
        if max_weight > 0.4:
            report += "⚠️ **Alta concentração**: Considere limitar exposição individual\n"
        
        report += "\n### Próximos Passos:\n"
        report += "1. Monitore regularmente os alfas (recalcular mensalmente)\n"
        report += "2. Rebalanceie o portfólio quando pesos desviarem >5%\n"
        report += "3. Atualize previsões V-Cov trimestralmente\n"
        report += "4. Considere hedging para posições de alto risco\n\n"
        
        # Disclaimers
        report += "## ⚖️ AVISOS LEGAIS\n\n"
        report += "- Esta análise é baseada em dados históricos e modelos preditivos\n"
        report += "- Performance passada não garante resultados futuros\n"
        report += "- Consulte um profissional qualificado antes de investir\n"
        report += "- Monitore constantemente os riscos do portfólio\n\n"
        
        return report
    
    def backtest_strategy(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str,
        rebalance_frequency: str = "quarterly"
    ) -> Dict:
        """
        Executa backtest da estratégia híbrida.
        
        Args:
            tickers: Lista de ativos
            start_date: Data início do backtest
            end_date: Data fim do backtest
            rebalance_frequency: Frequência de rebalanceamento
            
        Returns:
            Dict com resultados do backtest
        """
        # Implementação futura do backtest
        print("🔄 Funcionalidade de backtest será implementada em versão futura")
        return {
            'total_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0
        }


def example_usage():
    """Exemplo de uso do modelo híbrido."""
    
    print("🚀 Exemplo de uso do Modelo Híbrido V-Cov + Alfa")
    
    # Inicializar modelo
    model = HybridPortfolioModel(benchmark="^GSPC", risk_free_rate=0.02)
    
    # Definir restrições
    constraints = PortfolioConstraints(
        max_weight=0.3,
        min_weight=0.05,
        max_concentration=0.5,
        allow_short=False
    )
    
    # Executar análise híbrida
    tickers = "AAPL,GOOGL,MSFT,AMZN,TSLA"
    
    try:
        results = model.generate_portfolio_recommendation(
            tickers=tickers,
            vcov_period=2,
            vcov_window=60,
            alpha_period="2y",
            constraints=constraints,
            optimization_method="max_sharpe"
        )
        
        print("\n✅ Análise concluída!")
        print(f"📊 Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print(f"📈 Retorno Esperado: {results.expected_return:.2%}")
        print(f"📉 Volatilidade: {results.portfolio_volatility:.2%}")
        
        print("\n🎯 Pesos Otimizados:")
        for ticker, weight in results.optimal_weights.items():
            print(f"  {ticker}: {weight:.2%}")
        
        # Gerar relatório
        report = model.generate_detailed_report()
        print("\n📋 Relatório detalhado gerado!")
        
        return results, report
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        return None, None


if __name__ == "__main__":
    example_usage()
