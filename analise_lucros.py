"""
Análise de Lucros do Modelo Híbrido
Script simplificado para calcular lucros históricos.
"""

import os
import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# Configurar logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Adicionar diretório atual ao path
sys.path.append('/home/gabriel/itau_quant.ai')

from hybrid_portfolio_model import HybridPortfolioModel

def calcular_lucros_historicos():
    """Calcula lucros históricos do modelo híbrido."""
    print("💰 ANÁLISE DE LUCROS DO MODELO HÍBRIDO")
    print("=" * 50)
    
    # Configurações
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    capital_inicial = 100000.0  # $100k
    periodo_inicio = "2022-01-01"
    periodo_fim = "2024-01-01"
    
    print(f"📊 Configurações da Análise:")
    print(f"   - Ativos: {', '.join(tickers)}")
    print(f"   - Período: {periodo_inicio} a {periodo_fim}")
    print(f"   - Capital Inicial: ${capital_inicial:,.2f}")
    print()
    
    try:
        # Inicializar modelo híbrido
        print("🚀 Inicializando modelo híbrido...")
        modelo = HybridPortfolioModel()
        print("✅ Modelo híbrido inicializado!")
        print()
        
        # Baixar dados históricos
        print("📈 Baixando dados históricos...")
        dados = yf.download(
            tickers,
            start=periodo_inicio,
            end=periodo_fim,
            progress=False,
            auto_adjust=True
        )
        
        if len(tickers) == 1:
            precos = dados['Close'].to_frame()
            precos.columns = tickers
        else:
            precos = dados['Close']
        
        precos = precos.dropna()
        print(f"✅ Dados baixados: {len(precos)} dias de negociação")
        print()
        
        # Simular diferentes estratégias de rebalanceamento
        resultados = {}
        
        # 1. Estratégia Buy & Hold (pesos iguais)
        print("🧪 Testando estratégia Buy & Hold...")
        resultado_buy_hold = simular_buy_hold(precos, tickers, capital_inicial)
        resultados['Buy & Hold'] = resultado_buy_hold
        print(f"✅ Buy & Hold: Valor Final = ${resultado_buy_hold['valor_final']:,.2f}")
        print()
        
        # 2. Estratégia Híbrida com rebalanceamento mensal
        print("🧪 Testando estratégia Híbrida (rebalanceamento mensal)...")
        resultado_hibrido = simular_estrategia_hibrida(
            modelo, precos, tickers, capital_inicial, frequencia_dias=30
        )
        resultados['Híbrido Mensal'] = resultado_hibrido
        print(f"✅ Híbrido Mensal: Valor Final = ${resultado_hibrido['valor_final']:,.2f}")
        print()
        
        # 3. Estratégia Híbrida com rebalanceamento trimestral
        print("🧪 Testando estratégia Híbrida (rebalanceamento trimestral)...")
        resultado_hibrido_trim = simular_estrategia_hibrida(
            modelo, precos, tickers, capital_inicial, frequencia_dias=90
        )
        resultados['Híbrido Trimestral'] = resultado_hibrido_trim
        print(f"✅ Híbrido Trimestral: Valor Final = ${resultado_hibrido_trim['valor_final']:,.2f}")
        print()
        
        # Análise comparativa
        print("📊 ANÁLISE COMPARATIVA DE LUCROS")
        print("=" * 50)
        
        for estrategia, resultado in resultados.items():
            lucro = resultado['valor_final'] - capital_inicial
            rentabilidade = (resultado['valor_final'] / capital_inicial - 1) * 100
            
            print(f"🎯 {estrategia}:")
            print(f"   Valor Final: ${resultado['valor_final']:,.2f}")
            print(f"   Lucro/Prejuízo: ${lucro:,.2f}")
            print(f"   Rentabilidade: {rentabilidade:+.2f}%")
            print(f"   Volatilidade: {resultado['volatilidade']:.2f}%")
            print(f"   Sharpe: {resultado['sharpe']:.2f}")
            print()
        
        # Melhor estratégia
        melhor = max(resultados.items(), key=lambda x: x[1]['valor_final'])
        print(f"🏆 MELHOR ESTRATÉGIA: {melhor[0]}")
        print(f"   Valor Final: ${melhor[1]['valor_final']:,.2f}")
        print(f"   Lucro Total: ${melhor[1]['valor_final'] - capital_inicial:,.2f}")
        print(f"   Rentabilidade: {(melhor[1]['valor_final'] / capital_inicial - 1) * 100:+.2f}%")
        
        return resultados
        
    except Exception as e:
        print(f"❌ Erro na análise: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def simular_buy_hold(precos, tickers, capital_inicial):
    """Simula estratégia buy & hold com pesos iguais."""
    peso_por_ativo = 1.0 / len(tickers)
    
    # Preços inicial e final
    precos_inicial = precos.iloc[0]
    precos_final = precos.iloc[-1]
    
    # Calcular retornos
    retornos = precos.pct_change().dropna()
    retorno_portfolio = (retornos * peso_por_ativo).sum(axis=1)
    
    # Valor final
    retorno_total = (precos_final / precos_inicial - 1) * peso_por_ativo
    valor_final = capital_inicial * (1 + retorno_total.sum())
    
    # Métricas
    volatilidade = retorno_portfolio.std() * np.sqrt(252) * 100
    sharpe = retorno_portfolio.mean() / retorno_portfolio.std() * np.sqrt(252)
    
    return {
        'valor_final': valor_final,
        'volatilidade': volatilidade,
        'sharpe': sharpe,
        'retornos': retorno_portfolio
    }

def simular_estrategia_hibrida(modelo, precos, tickers, capital_inicial, frequencia_dias=30):
    """Simula estratégia híbrida com rebalanceamento periódico."""
    
    # Datas de rebalanceamento
    datas_rebalanceamento = []
    data_atual = precos.index[0]
    
    while data_atual <= precos.index[-1]:
        # Encontrar próxima data disponível
        datas_disponiveis = precos.index[precos.index >= data_atual]
        if len(datas_disponiveis) > 0:
            datas_rebalanceamento.append(datas_disponiveis[0])
        data_atual += timedelta(days=frequencia_dias)
    
    valor_portfolio = capital_inicial
    pesos_atuais = {ticker: 1.0/len(tickers) for ticker in tickers}  # Começar com pesos iguais
    valores_historicos = []
    
    for i, data in enumerate(datas_rebalanceamento):
        if data not in precos.index:
            continue
            
        print(f"🔄 Rebalanceamento {i+1}: {data.strftime('%Y-%m-%d')}")
        
        try:
            # Gerar nova recomendação do modelo híbrido
            if i > 0:  # Pular primeira iteração (usar pesos iguais)
                tickers_str = ','.join(tickers)
                resultado = modelo.generate_portfolio_recommendation(
                    tickers=tickers_str,
                    optimization_method='risk_parity',  # Usar risk parity
                    vcov_window=60,
                    alpha_period='2y'
                )
                
                if resultado and hasattr(resultado, 'optimal_weights'):
                    novos_pesos = resultado.optimal_weights
                    # Normalizar pesos
                    total_peso = sum(novos_pesos.values())
                    if total_peso > 0:
                        pesos_atuais = {
                            ticker: peso / total_peso 
                            for ticker, peso in novos_pesos.items()
                        }
            
            # Simular performance até próximo rebalanceamento
            if i < len(datas_rebalanceamento) - 1:
                data_fim = datas_rebalanceamento[i + 1]
            else:
                data_fim = precos.index[-1]
            
            # Obter dados do período
            mask = (precos.index >= data) & (precos.index <= data_fim)
            precos_periodo = precos.loc[mask]
            
            if len(precos_periodo) > 1:
                retornos_periodo = precos_periodo.pct_change().dropna()
                
                # Calcular retorno do portfolio
                for _, retornos_dia in retornos_periodo.iterrows():
                    retorno_portfolio = sum(
                        pesos_atuais.get(ticker, 0) * retornos_dia[ticker]
                        for ticker in tickers if ticker in retornos_dia.index
                    )
                    valor_portfolio *= (1 + retorno_portfolio)
                    valores_historicos.append(valor_portfolio)
        
        except Exception as e:
            print(f"⚠️ Erro no rebalanceamento {i+1}: {str(e)}")
            # Continuar com pesos atuais em caso de erro
    
    # Calcular métricas finais
    if len(valores_historicos) > 1:
        valores_series = pd.Series(valores_historicos)
        retornos_portfolio = valores_series.pct_change().dropna()
        
        volatilidade = retornos_portfolio.std() * np.sqrt(252) * 100
        sharpe = retornos_portfolio.mean() / retornos_portfolio.std() * np.sqrt(252) if retornos_portfolio.std() > 0 else 0
    else:
        volatilidade = 0
        sharpe = 0
        retornos_portfolio = pd.Series([0])
    
    return {
        'valor_final': valor_portfolio,
        'volatilidade': volatilidade,
        'sharpe': sharpe,
        'retornos': retornos_portfolio
    }

if __name__ == "__main__":
    print(f"📅 Análise iniciada em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    resultados = calcular_lucros_historicos()
    
    print("\n" + "=" * 50)
    print("🎉 ANÁLISE DE LUCROS CONCLUÍDA!")
    print("=" * 50)
    
    if resultados:
        print("✅ Análise executada com sucesso!")
        print("📊 Consulte os resultados acima para detalhes dos lucros.")
    else:
        print("❌ Falha na execução da análise.")
    
    print(f"\n📅 Análise concluída em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
