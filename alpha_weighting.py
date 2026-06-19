"""
Módulo de Ponderação Alfa (Alpha Weighting)
==========================================

Este módulo implementa técnicas de ponderação baseadas em alfa para otimização de portfólios.
Calcula o alfa (retorno em excesso) de cada ativo e determina pesos baseados na capacidade
de geração de alfa de cada ativo.

Author: Gabriel
Date: 2025-09-07
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class AlphaWeighting:
    """
    Classe para cálculo de ponderação baseada em alfa.
    
    Metodologia:
    1. Calcula beta de cada ativo contra benchmark
    2. Calcula alfa usando CAPM
    3. Determina pesos baseados nos alfas
    """
    
    def __init__(self, benchmark: str = "^GSPC", risk_free_rate: float = 0.02):
        """
        Inicializa o calculador de ponderação alfa.
        
        Args:
            benchmark: Símbolo do benchmark (padrão: S&P 500)
            risk_free_rate: Taxa livre de risco anual (padrão: 2%)
        """
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate / 252  # Converter para diário
        self.alphas = {}
        self.betas = {}
        self.weights = {}
        
    def fetch_data(self, symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Baixa dados históricos dos ativos e benchmark.
        
        Args:
            symbols: Lista de símbolos dos ativos
            period: Período de dados (padrão: 2 anos)
            
        Returns:
            Dict com DataFrames dos preços
        """
        print(f"📊 Baixando dados para {len(symbols)} ativos...")
        
        data = {}
        all_symbols = symbols + [self.benchmark]
        
        try:
            # Baixar todos os dados de uma vez
            raw_data = yf.download(all_symbols, period=period, progress=False, auto_adjust=True)
            
            # Verificar se os dados foram baixados com sucesso
            if raw_data is None or raw_data.empty:
                print(f"❌ Erro: Nenhum dado foi baixado")
                return {}
            
            # Com auto_adjust=True, usar coluna 'Close' (já ajustada)
            if 'Close' not in raw_data.columns.get_level_values(0):
                print(f"❌ Erro: Coluna 'Close' não encontrada nos dados")
                return {}
            
            # Extrair preços ajustados (Close com auto_adjust=True equivale ao antigo Adj Close)
            close_data = raw_data['Close']
            
            if isinstance(close_data, pd.Series):
                # Caso seja apenas um ativo
                data[all_symbols[0]] = close_data.to_frame()
            else:
                for symbol in all_symbols:
                    if symbol in close_data.columns:
                        symbol_data = close_data[symbol].dropna()
                        if not symbol_data.empty:
                            data[symbol] = symbol_data
                        else:
                            print(f"⚠️ Dados vazios para {symbol}")
                    else:
                        print(f"⚠️ Símbolo {symbol} não encontrado nos dados")
                    
        except KeyError as e:
            print(f"❌ Erro ao baixar dados: {e}")
            print("Possível causa: ticker inválido ou dados indisponíveis")
            return {}
        except Exception as e:
            print(f"❌ Erro inesperado ao baixar dados: {e}")
            return {}
            
        print(f"✅ Dados baixados com sucesso!")
        return data
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calcula retornos logarítmicos.
        
        Args:
            prices: Série de preços
            
        Returns:
            Série de retornos
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_beta_alpha(self, asset_returns: pd.Series, market_returns: pd.Series) -> Tuple[float, float]:
        """
        Calcula beta e alfa de um ativo usando regressão linear.
        
        Args:
            asset_returns: Retornos do ativo
            market_returns: Retornos do mercado
            
        Returns:
            Tupla (beta, alfa)
        """
        # Alinhar as séries
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:  # Mínimo de observações
            return 0.0, 0.0
            
        y = aligned_data.iloc[:, 0].values  # Retornos do ativo
        x = aligned_data.iloc[:, 1].values  # Retornos do mercado
        
        # Regressão linear: R_asset = α + β * R_market + ε
        x_reshaped = x.reshape(-1, 1)
        model = LinearRegression().fit(x_reshaped, y)
        
        beta = model.coef_[0]
        alpha_intercept = model.intercept_
        
        # Converter alfa para anualizado
        alpha_annual = alpha_intercept * 252
        
        return beta, alpha_annual
    
    def calculate_portfolio_weights(self, symbols: List[str], period: str = "2y") -> Dict[str, float]:
        """
        Calcula pesos do portfólio baseados em alfa.
        
        Args:
            symbols: Lista de símbolos dos ativos
            period: Período de análise
            
        Returns:
            Dicionário com pesos dos ativos
        """
        print(f"🔄 Calculando ponderação alfa para {len(symbols)} ativos...")
        
        # Baixar dados
        data = self.fetch_data(symbols, period)
        
        if not data or self.benchmark not in data:
            print("❌ Erro: Dados insuficientes")
            return {symbol: 1.0/len(symbols) for symbol in symbols}  # Peso igual
        
        # Calcular retornos do benchmark
        market_prices = data[self.benchmark]
        market_returns = self.calculate_returns(market_prices)
        
        alphas = []
        valid_symbols = []
        
        # Calcular alfa para cada ativo
        for symbol in symbols:
            if symbol in data:
                asset_prices = data[symbol]
                asset_returns = self.calculate_returns(asset_prices)
                
                beta, alpha = self.calculate_beta_alpha(asset_returns, market_returns)
                
                self.betas[symbol] = beta
                self.alphas[symbol] = alpha
                
                print(f"📈 {symbol}: α = {alpha:.4f}, β = {beta:.4f}")
                
                alphas.append(alpha)
                valid_symbols.append(symbol)
            else:
                print(f"⚠️ Dados não encontrados para {symbol}")
        
        if not alphas:
            print("❌ Nenhum alfa calculado")
            return {symbol: 1.0/len(symbols) for symbol in symbols}
        
        # Converter alfas negativos para positivos para ponderação
        # Estratégia: somar constante para tornar todos positivos
        min_alpha = min(alphas)
        if min_alpha < 0:
            adjusted_alphas = [alpha - min_alpha + 0.001 for alpha in alphas]
        else:
            adjusted_alphas = [max(alpha, 0.001) for alpha in alphas]  # Evitar divisão por zero
        
        # Calcular pesos proporcionais aos alfas
        total_alpha = sum(adjusted_alphas)
        weights = {}
        
        for i, symbol in enumerate(valid_symbols):
            weight = adjusted_alphas[i] / total_alpha
            weights[symbol] = weight
            self.weights[symbol] = weight
        
        # Preencher símbolos faltantes com peso zero
        for symbol in symbols:
            if symbol not in weights:
                weights[symbol] = 0.0
        
        print(f"✅ Ponderação alfa calculada!")
        return weights
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """
        Retorna estatísticas do portfólio.
        
        Returns:
            Dicionário com estatísticas
        """
        if not self.alphas:
            return {}
        
        total_alpha = sum(self.alphas.values())
        avg_beta = np.mean(list(self.betas.values()))
        
        return {
            'total_alpha': total_alpha,
            'average_beta': avg_beta,
            'num_assets': len(self.alphas),
            'alpha_concentration': max(self.weights.values()) if self.weights else 0
        }
    
    def print_summary(self):
        """Imprime resumo da análise alfa."""
        if not self.alphas:
            print("❌ Nenhuma análise disponível")
            return
        
        print("\n" + "="*50)
        print("📊 RESUMO DA PONDERAÇÃO ALFA")
        print("="*50)
        
        stats = self.get_portfolio_stats()
        
        print(f"🎯 Alfa Total do Portfólio: {stats.get('total_alpha', 0):.4f}")
        print(f"📈 Beta Médio: {stats.get('average_beta', 0):.4f}")
        print(f"📊 Número de Ativos: {stats.get('num_assets', 0)}")
        print(f"⚖️ Concentração Máxima: {stats.get('alpha_concentration', 0):.2%}")
        
        print("\n📈 DETALHES POR ATIVO:")
        print("-" * 40)
        
        for symbol in self.alphas.keys():
            alpha = self.alphas[symbol]
            beta = self.betas[symbol]
            weight = self.weights.get(symbol, 0)
            
            alpha_status = "🟢" if alpha > 0 else "🔴"
            
            print(f"{alpha_status} {symbol:8} | Peso: {weight:6.2%} | α: {alpha:8.4f} | β: {beta:6.4f}")


def example_usage():
    """Exemplo de uso da ponderação alfa."""
    print("🚀 Exemplo de Ponderação Alfa")
    
    # Símbolos para análise
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    # Criar calculador
    alpha_calc = AlphaWeighting(benchmark="^GSPC", risk_free_rate=0.02)
    
    # Calcular pesos
    weights = alpha_calc.calculate_portfolio_weights(symbols)
    
    # Mostrar resultados
    alpha_calc.print_summary()
    
    print(f"\n🎯 PESOS FINAIS DO PORTFÓLIO:")
    for symbol, weight in weights.items():
        print(f"   {symbol}: {weight:.2%}")


if __name__ == "__main__":
    example_usage()
