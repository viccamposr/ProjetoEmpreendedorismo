# EcoLetras — PoC com IA local (sem API paga)

## O que o projeto usa
- **Ollama + Qwen 2.5 3B:** gera textos ambientais localmente, sem chave de API.
- **faster-whisper:** transcreve a leitura em português no próprio computador.
- **Gradio:** cria a interface web local.

Não há cobrança por requisição de API. Ollama e o modelo são gratuitos para baixar e executar. É necessário ter espaço em disco, memória RAM e conexão com a internet para baixar os programas/modelos inicialmente.

## 1. Instalar o Ollama (Windows)
1. Acesse https://ollama.com/download e baixe a versão para Windows.
2. Instale e abra o Ollama.
3. Abra um novo Prompt de Comando e execute:

```cmd
ollama pull qwen2.5:3b
```

Aguarde o download do modelo. Para testar, execute `ollama run qwen2.5:3b`. Digite uma pergunta; para sair, digite `/bye`.

## 2. Abrir a pasta do projeto no terminal
Entre na pasta onde estão este README e o arquivo `app.py`. Exemplo para este computador:

```cmd
cd /d "C:\Users\victo\OneDrive\Área de Trabalho\Projeto IA"
```

## 3. Instalar dependências Python
Recomendado criar um ambiente virtual:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Se o ambiente virtual já estiver ativo, basta executar a última linha.

## 4. Executar o EcoLetras
Confirme que o Ollama está aberto e execute:

```cmd
python app.py
```

Abra no navegador o endereço local mostrado no terminal, normalmente http://127.0.0.1:7860.

## Observações
- Na primeira execução, o faster-whisper baixa o modelo de reconhecimento de fala. Isso também não gera cobrança de API.
- O Qwen 2.5 3B pode demorar para gerar texto em computadores mais simples.
- O percentual mostrado é uma comparação aproximada entre as palavras do texto e a transcrição automática. Não mede fonemas nem comprova pronúncia correta.
- O reconhecimento pode errar com vozes infantis, sotaques e ruídos. Use a ferramenta com acompanhamento de um adulto/educador; não trate o percentual como nota ou diagnóstico.
- O código não salva os arquivos de áudio em banco de dados. O processamento ocorre durante a atividade.
