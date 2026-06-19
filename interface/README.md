# Interface Module

Esta pasta contém todos os arquivos relacionados à interface do usuário do sistema de otimização de portfólio.

## 📁 Arquivos

### 🚀 Interface Principal
- **`gradio_interface.py`** - Interface principal do sistema (versão híbrida simplificada)
- **`__init__.py`** - Arquivo de inicialização do módulo

### 📚 Versões de Backup
- **`gradio_interface_backup.py`** - Backup da interface completa original
- **`gradio_interface_old.py`** - Versão anterior da interface
- **`gradio_interface_simple.py`** - Versão simplificada para testes

## 🎯 Interface Ativa

A interface atualmente em uso é a **`gradio_interface.py`**, que contém:

- ✅ Dashboard do modelo híbrido
- ✅ Análise LSTM + Alpha
- ✅ Visualizações interativas
- ✅ Interface limpa e intuitiva

## 🚀 Como usar

```python
from interface.gradio_interface import GradioInterface

# Criar interface
interface = GradioInterface()

# Lançar aplicação
interface.launch()
```

## 📊 Funcionalidades

- **Previsão V-Cov** com LSTM
- **Cálculo de Alpha** usando CAPM
- **Otimização de portfólio** (Risk Parity, Alpha Weighted, etc.)
- **Visualizações** em tempo real
- **Análise de performance** completa
