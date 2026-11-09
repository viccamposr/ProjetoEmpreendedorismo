# EcoLeitura

Site Python/Gradio para gerar atividades ambientais e avaliar uma leitura em português brasileiro.

## O que mudou

O protótipo original tinha frases fixas e retornava uma mensagem de sucesso sem processar a voz. Agora a geração por IA usa Gemini e a avaliação usa Azure Speech Pronunciation Assessment, com o texto como referência. Os exemplos locais continuam disponíveis e estão identificados como textos pré-escritos.

O relatório mostra a nota geral do modelo, precisão dos sons, fluência, completude, texto reconhecido e notas por palavra. Palavras omitidas não recebem uma nota inventada. A WER é calculada separadamente por distância de edição entre palavras, preservando os acentos e ignorando caixa e pontuação; ela pode superar 100% quando há muitas inserções. A transcrição do avaliador é condicionada pelo texto de referência: essa WER é auxiliar, não uma avaliação independente do reconhecedor.

## Executar no Windows

Use Python 3.11 ou superior, com suporte às dependências científicas. Em um terminal dentro desta pasta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edite .env e preencha as chaves antes de iniciar.
.\.venv\Scripts\python.exe projeto.py
```

Se `python` abrir a Microsoft Store, use `py` ou o caminho completo de uma instalação Python. Não é necessário ativar a venv. O navegador abre em um endereço local, normalmente http://127.0.0.1:7860. Permita o acesso ao microfone. Também é possível enviar um WAV.

Se a instalação de PyYAML exigir Visual C++ em versões recentes do Python, é possível instalar sua implementação em Python, sem a extensão C:

```powershell
$env:PYYAML_FORCE_LIBYAML = '0'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Configuração

- `GEMINI_API_KEY`: chave da Gemini API no Google AI Studio; necessária apenas para gerar textos por IA.
- `GEMINI_MODEL`: modelo disponível na sua conta; padrão `gemini-3.5-flash-lite`. Pode ser trocado sem editar o código. Modelos antigos podem aparecer na listagem da API mesmo sem acesso à geração para novas contas. Se receber HTTP 404, confira o acesso ao modelo no Google AI Studio, atualize esta variável no `.env` e reinicie o aplicativo.
- `AZURE_SPEECH_KEY`: chave de um recurso Azure Speech.
- `AZURE_SPEECH_REGION`: região do mesmo recurso, por exemplo `brazilsouth`.

As chaves ficam no servidor, em `.env` (ignorado pelo Git). Não as coloque no código ou em capturas de tela. Os provedores podem cobrar por uso. Sem chaves é possível explorar os exemplos, mas não obter notas de pronúncia. Falhas da API não são substituídas silenciosamente por resultados simulados.

## Usar

1. Escolha tema, nível e origem do texto; clique em **Criar atividade**.
2. Leia o texto e grave até 30 segundos em um local silencioso.
3. Autorize o envio da gravação para avaliação e clique em **Avaliar minha leitura**.
4. Revise a transcrição, as notas e as palavras sinalizadas; pratique e grave novamente.

Uma atividade nova limpa o áudio e os resultados anteriores. Uma avaliação que falha limpa as notas anteriores. O áudio é validado e convertido em memória para WAV PCM de 16 bits, mono, 16 kHz. Gravações acima do limite são recusadas, sem truncamento silencioso. Textos são curtos para caber no limite, mas o tempo de leitura de cada pessoa pode variar.

## Estrutura e testes

- `projeto.py`: interface e eventos Gradio, sem iniciar servidor durante importação.
- `servicos.py`: integrações HTTP, validações, áudio, métricas e relatório.
- `test_servicos.py`: testes com respostas de API simuladas e áudio sintético; não usam credenciais nem enviam dados.

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Testes locais verificam a integração construída, não a qualidade pedagógica do modelo. Antes de usar com alunos, valide com gravações autorizadas e avaliações humanas, incluindo diferentes sotaques, idades, microfones e condições de ruído. As notas são estimativas do serviço, não um percentual objetivo de pronúncia correta. O sistema não implementa diagnóstico clínico nem avaliação de prosódia em português.

Verificação realizada em 05/11/2026: 12 testes passaram com Python 3.15.2 e Gradio 6.31.0; a interface foi criada e o servidor respondeu HTTP 200 em localhost. O servidor de teste foi encerrado. As chamadas reais ao Gemini e ao Azure não foram exercitadas, pois não foram fornecidas credenciais nem gravações autorizadas.

Verificação posterior em 05/11/2026: após configurar o Gemini, a geração real foi validada com `gemini-3.5-flash-lite` e 16 testes locais passaram. O modelo anterior retornava HTTP 404; o Google [informa restrições de acesso ao Gemini 2.5 para novos projetos](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash). Respostas maiores que o limite podem ser reduzidas preservando apenas frases completas, sem cortar palavras. Falhas de geração são exibidas no status e preservam a atividade anterior. A avaliação Azure continua sem validação real.

## Dados e publicação

No modo IA, tema e nível vão ao Google; na avaliação, áudio e texto vão à Microsoft. A interface pede autorização antes desse envio. Isso é um controle do protótipo e não substitui um processo institucional de autorização e proteção de dados. Não há banco de dados ou histórico próprio. O Gradio cria arquivos temporários e está configurado para remover arquivos com mais de uma hora em ciclos de cinco minutos enquanto a aplicação estiver ativa; a retenção nos provedores segue as condições contratadas.

O servidor inicia apenas em `127.0.0.1`, sem link público. Para publicar, serão necessários HTTPS (para o microfone remoto), autenticação, controle de uso/custos, gestão de segredos e revisão da retenção dos dados. Esta alteração prepara o protótipo local; não publica o site.

## Referências das integrações

- [Gemini: generateContent](https://ai.google.dev/api/generate-content)
- [Azure: avaliação de pronúncia](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment)
- [Azure: REST para áudio curto e formato de resposta](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text-short)
- [Azure: idiomas de avaliação](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=pronunciation-assessment)
- [Gradio 6: migração dos parâmetros de tema](https://gradio.app/guides/gradio-6-migration-guide)
