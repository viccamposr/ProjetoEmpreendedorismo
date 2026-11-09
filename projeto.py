"""
EcoLeitura — PoC de alfabetização e educação ambiental
Arquivo único: projeto.py

Funcionalidades:
1) Geração de texto ambiental com LLM local (Ollama).
2) Gravação/envio de áudio e transcrição em português (faster-whisper).
3) Comparação entre texto esperado e transcrição, com feedback estimado.
4) Relatório da atividade na própria página.

IMPORTANTE:
- A comparação textual NÃO mede fonemas com precisão. O resultado é uma estimativa
  de correspondência entre o que foi proposto e o que o reconhecedor transcreveu.
- Para uso com crianças, faça testes supervisionados, evite guardar áudios e não
  trate a pontuação como nota ou diagnóstico.
"""

import os
import json
import urllib.request
import urllib.error
import re
import unicodedata
import random
from datetime import datetime

import gradio as gr
from faster_whisper import WhisperModel


# ----------------------------- CONFIGURAÇÕES -----------------------------

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

TEMAS = [
    "Reciclagem e separação do lixo",
    "Mudanças climáticas",
    "Desmatamento e queimadas",
    "Extinção de espécies",
    "Água e uso consciente",
    "Ciclo da vida e as plantas",
    "Animais e seus lares",
    "Economia de energia",
    "Consumo consciente e reaproveitamento",
    "Poluição e cuidado com a natureza",
]


SUBTEMAS = {
    "Reciclagem e separação do lixo": [
        "Como o papel é reciclado",
        "Como o vidro é reciclado",
        "Como o plástico é reciclado",
        "Como separar os tipos de lixo",
        "O que é coleta seletiva",
        "Para que servem as lixeiras de coleta seletiva"
    ],

    "Mudanças climáticas": [
        "O que são mudanças climáticas",
        "Por que a Terra está ficando mais quente",
        "Como o calor afeta os animais",
        "Como as mudanças climáticas afetam as chuvas",
        "Por que as geleiras estão derretendo",
        "Como reduzir a emissão de gases que aquecem a Terra"
    ],

    "Desmatamento e queimadas": [
        "O que é desmatamento",
        "Por que as queimadas prejudicam as florestas",
        "Como o fogo afeta os animais",
        "Por que as árvores ajudam a proteger o solo",
        "O que acontece quando uma floresta é destruída",
        "Como podemos proteger as florestas"
    ],

    "Extinção de espécies": [
        "O que significa extinção de espécies",
        "Por que alguns animais estão ameaçados",
        "Como a perda de florestas afeta os animais",
        "Por que devemos proteger animais ameaçados",
        "Como a poluição prejudica os animais",
        "Como proteger os lugares onde os animais vivem"
    ],

    "Água e uso consciente": [
        "Por que precisamos de água",
        "Como economizar água ao escovar os dentes",
        "Por que devemos fechar a torneira",
        "Como evitar o desperdício de água",
        "Por que precisamos de água limpa",
        "Como a água é usada pelas plantas"
    ],

    "Ciclo da vida e as plantas": [
        "Como uma semente germina",
        "O que as plantas precisam para crescer",
        "Como surgem as raízes",
        "Para que servem as folhas",
        "Como as flores podem formar frutos",
        "Como os frutos ajudam a espalhar sementes"
    ],

    "Animais e seus lares": [
        "Onde os pássaros fazem seus ninhos",
        "Como os peixes vivem na água",
        "Por que as abelhas vivem em colmeias",
        "Como os animais encontram abrigo",
        "Por que as florestas são importantes para os animais",
        "Como os animais se protegem em seus ambientes"
    ],

    "Economia de energia": [
        "Por que devemos apagar as luzes",
        "Como aproveitar a luz do sol",
        "Por que desligar aparelhos sem uso",
        "Como economizar energia em casa",
        "Como a energia elétrica chega às casas",
        "Por que evitar o desperdício de energia"
    ],

    "Consumo consciente e reaproveitamento": [
        "Por que devemos evitar compras desnecessárias",
        "Como reutilizar potes e embalagens",
        "Por que devemos cuidar dos nossos objetos",
        "Como aproveitar os dois lados do papel",
        "Por que evitar o desperdício de alimentos",
        "Como consertar objetos em vez de descartá-los"
    ],

    "Poluição e cuidado com a natureza": [
        "O que é poluição da água",
        "Como o lixo pode poluir os rios",
        "O que causa a poluição do ar",
        "Por que não devemos jogar lixo no chão",
        "Como a poluição afeta os animais",
        "Como manter os espaços naturais limpos"
    ]
}

indice_subtemas = {}


NIVEIS = ["Iniciante", "Intermediário", "Avançado"]

MASCOTES = {
    "Tuca": "a Tartaruga",
    "Pintado": "a Onça-pintada",
    "Lia": "a Raposa",
    "Capiz": "a Capivara",
    "Azul": "a Arara-azul",
    "Lento": "o Bicho-preguiça",
    "Tuti": "o Tamanduá",
    "Pipo": "o Sapo"
}

PASTA_MASCOTES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mascotes"
)

IMAGENS_MASCOTES = [
    (
        os.path.join(PASTA_MASCOTES, f"{nome.lower()}.png"),
        f"{nome}, {animal}"
    )
    for nome, animal in MASCOTES.items()
]

# Modelos carregados sob demanda para a primeira abertura da aplicação ser mais rápida.
_whisper_model = None


# ----------------------------- FUNÇÕES AUXILIARES -----------------------------

def normalizar(texto: str) -> str:
    """Padroniza texto para comparação simples, sem diferenciar acentos ou caixa."""
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def gerar_texto(tema, nivel, historico):
    limites = {
        "Iniciante": (5, 10, 100),
        "Intermediário": (10, 20, 150),
        "Avançado": (20, 40, 210),
    }

    minimo, maximo, limite_tokens = limites[nivel]

    palavras_revisao = []

    for atividade in (historico or [])[-3:]:
        palavras_revisao.extend(
            atividade.get("palavras_revisao", [])
        )

    palavras_revisao = list(dict.fromkeys(palavras_revisao))[:3]


    if tema in SUBTEMAS:
        opcoes = SUBTEMAS[tema]

        chave = (tema, nivel)
        indice = indice_subtemas.get(chave, 0)

        subtema = opcoes[indice % len(opcoes)]

        indice_subtemas[chave] = indice + 1
    else:
        subtema = tema
    

    instrucoes_nivel = {
        "Iniciante": """
        Escreva no máximo 2 frases curtas.
        Cada frase deve ter de 5 a 9 palavras.
        Explique apenas uma informação simples.
        A segunda frase deve completar a primeira.
        Use palavras comuns e fáceis de ler.
        Evite termos científicos e frases complexas.
        """,

        "Intermediário": """
        Escreva no máximo 3 frases curtas e conectadas.
        Use palavras comuns e algumas palavras novas.
        Cada frase deve ter entre 8 e 12 palavras.
        Explique o tema com um pouco mais de detalhes.
        Use vocabulário cotidiano.
        """,

        "Avançado": """
        Escreva no máximo 5 frases conectadas.
        Use vocabulário um pouco mais elaborado.
        Cada frase deve ter entre 8 e 15 palavras.
        Desenvolva melhor o assunto.
        """
    }

    prompt = f"""
        Você é um professor de educação ambiental que cria
        pequenos textos informativos para alfabetização infantil.

        Sua tarefa é explicar um conhecimento verdadeiro
        sobre o meio ambiente usando linguagem simples.

        Tema principal: {tema}
        Assunto específico: {subtema}
        Nível de leitura: {nivel}

        Explique SOMENTE o assunto específico indicado.
        Não faça uma explicação genérica sobre o tema principal.
        Ensine uma informação concreta, verdadeira e simples.

        REGRAS DO NÍVEL:
        {instrucoes_nivel[nivel]}

        TIPO DE TEXTO:
        - Escreva um texto INFORMATIVO e EXPLICATIVO.
        - Ensine uma informação concreta sobre o tema.
        - Explique o que é, como funciona ou por que acontece.
        - Escolha apenas uma informação principal para explicar.
        - A segunda frase deve complementar a primeira.
        - Use informações corretas e fáceis de compreender.
        - Escreva como um pequeno trecho de livro didático.

        REGRAS DE ESCRITA:
        - Use frases completas, naturais e com sentido.
        - Use palavras simples e adequadas ao nível.
        - Evite palavras abstratas ou técnicas desnecessárias.
        - Não crie histórias, personagens ou acontecimentos.
        - Não invente informações.
        - Não escreva frases sem relação entre si.
        - Não use frases poéticas ou metafóricas.
        - Não use frases genéricas como "O planeta é bonito".
        - Não use comandos como "Vamos cuidar da natureza".
        - Não escreva título, perguntas ou listas.
        - Escreva somente um parágrafo.
        - Respeite a quantidade de frases definida para o nível.

        REGRAS DE COERÊNCIA:
        - Todas as frases devem explicar a mesma ideia.
        - Não misture diferentes assuntos no mesmo texto.
        - Não use palavras apenas para completar o tamanho.
        - Evite informações vagas ou sem explicação.
        - Priorize clareza, correção e sentido.

        Palavras para revisar:
        {", ".join(palavras_revisao) if palavras_revisao else "Nenhuma"}

        Se houver palavras para revisar, utilize apenas aquelas
        que combinarem naturalmente com o tema e o nível.
        Nunca prejudique o sentido do texto para incluí-las.

        Antes de responder, confira:
        1. O texto realmente ensina algo sobre o tema?
        2. Todas as informações estão corretas?
        3. As frases estão conectadas?
        4. A linguagem está adequada ao nível?
        5. A quantidade de frases está correta?

        Responda SOMENTE com o texto final.
        """

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": limite_tokens,
            "num_ctx": 2048
        }
    }).encode("utf-8")

    requisicao = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(requisicao, timeout=300) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))

        texto = dados.get("response", "").strip()
        quantidade_frases = {
            "Iniciante": 2,
            "Intermediário": 3,
            "Avançado": 5
        }

        frases = re.split(r'(?<=[.!?])\s+', texto)

        texto = " ".join(
            frases[:quantidade_frases[nivel]]
        )

        if not texto:
            raise ValueError("O modelo não retornou um texto.")

        return texto

    except Exception as erro:
        raise gr.Error(f"Erro ao gerar texto: {erro}")

def obter_modelo_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def transcrever_audio(caminho_audio):
    if not caminho_audio:
        raise gr.Error("Grave sua leitura ou envie um arquivo de áudio.")

    modelo = obter_modelo_whisper()
    segmentos, info = modelo.transcribe(
        caminho_audio,
        language="pt",
        beam_size=5,
        vad_filter=True,
    )
    transcricao = " ".join(segmento.text.strip() for segmento in segmentos).strip()

    if not transcricao:
        raise gr.Error(
            "Não foi possível reconhecer a fala. Tente gravar novamente em um "
            "ambiente silencioso e fale em um ritmo natural."
        )

    return transcricao


def avaliar_leitura(texto_original, transcricao):
    """
    Faz comparação aproximada por palavras.
    Não é uma avaliação fonética/fonêmica nem um diagnóstico de fala.
    """
    palavras_originais = normalizar(texto_original).split()
    palavras_lidas = normalizar(transcricao).split()

    if not palavras_originais:
        raise gr.Error("Gere um texto antes de avaliar a leitura.")

    # Alinhamento simples para identificar palavras iguais, omitidas e substituídas.
    import difflib

    comparador = difflib.SequenceMatcher(
        a=palavras_originais,
        b=palavras_lidas,
        autojunk=False,
    )

    corretas = []
    diferentes = []
    faltantes = []
    adicionadas = []

    for operacao, i1, i2, j1, j2 in comparador.get_opcodes():
        if operacao == "equal":
            corretas.extend(palavras_originais[i1:i2])
        elif operacao == "replace":
            pares = min(i2 - i1, j2 - j1)
            for indice in range(pares):
                diferentes.append(
                    (palavras_originais[i1 + indice], palavras_lidas[j1 + indice])
                )
            faltantes.extend(palavras_originais[i1 + pares:i2])
            adicionadas.extend(palavras_lidas[j1 + pares:j2])
        elif operacao == "delete":
            faltantes.extend(palavras_originais[i1:i2])
        elif operacao == "insert":
            adicionadas.extend(palavras_lidas[j1:j2])

    total = len(palavras_originais)
    percentual = round((len(corretas) / total) * 100) if total else 0

    if percentual >= 90:
        mensagem = "Ótimo trabalho! A transcrição ficou bem próxima do texto."
    elif percentual >= 70:
        mensagem = "Muito bem! Você está aprendendo. Vamos praticar mais algumas palavras."
    else:
        mensagem = "Continue tentando! Leia devagar e grave novamente com calma."

    linhas = []
    if diferentes:
        linhas.extend([f"{esperada} → reconhecido como {reconhecida}"
                       for esperada, reconhecida in diferentes])
    if faltantes:
        linhas.append("Possivelmente não identificadas: " + ", ".join(faltantes))
    if adicionadas:
        linhas.append("Palavras extras reconhecidas: " + ", ".join(adicionadas))

    detalhes = "\n".join(linhas) if linhas else "Nenhuma diferença de palavras foi identificada."

    palavras_revisao = (
    [palavra for palavra, _ in diferentes] + faltantes
    )

    return percentual, mensagem, detalhes, palavras_revisao


# ----------------------------- EVENTOS DA INTERFACE -----------------------------

def gerar_e_atualizar(tema, nivel, historico):
    texto = gerar_texto(tema, nivel, historico)
    return texto, "Texto gerado! Agora leia em voz alta e grave sua leitura."


def processar_leitura(audio, texto_original, tema, nivel, historico, nome_mascote):
    if not texto_original or not texto_original.strip():
        raise gr.Error("Primeiro, gere um texto ambiental.")

    transcricao = transcrever_audio(audio)
    percentual, mensagem, detalhes, palavras_revisao = avaliar_leitura(
    texto_original, transcricao
    )

    registro = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "tema": tema,
        "nivel": nivel,
        "resultado": f"{percentual}%",
        "mensagem": mensagem,
        "transcricao": transcricao,
        "palavras_revisao": palavras_revisao,
    }

    historico = (historico or []) + [registro]

    relatorio = f"""
       RELATÓRIO DE LEITURA

        TEMA: {tema}  
        NÍVEL: {nivel}  
        CORRESPONDÊNCIA ESTIMADA: {percentual}%  

        FEEDBACK: {mensagem}


        PALAVRAS PARA REVISAR:

        {detalhes}

        --> Esta porcentagem representa a correspondência aproximada entre as palavras do texto
        --> e a transcrição automática. Ela não comprova a pronúncia correta de cada som.
    """
    tabela = [
        [
            item["data"],
            item["tema"],
            item["nivel"],
            item["resultado"],
            item["mensagem"],
        ]
        for item in historico
    ]
    imagem_mascote, mensagem_mascote = reacao_mascote(
        nome_mascote,
        percentual
    )

    return ( transcricao, relatorio, tabela, historico,
        gr.update(
            value=imagem_mascote,
            visible=imagem_mascote is not None
        ),
        mensagem_mascote
    )


def limpar_atividade():
    return "", None, "", "", [], [], gr.update(value=None, visible=False), ""

def escolher_mascote(selecao):
    if selecao is None:
        raise gr.Error("Escolha um mascote para continuar.")

    nome = list(MASCOTES.keys())[selecao]

    animal = MASCOTES[nome]

    mensagem = (
        f"### Seu companheiro é {nome}!\n\n"
        f"{nome}, {animal.lower()}, vai acompanhar "
        "você nas aventuras de leitura!"
    )

    return nome, mensagem

def reacao_mascote(nome_mascote, percentual):
    if not nome_mascote:
        return None, "Escolha um mascote para acompanhar sua aventura!"

    if percentual >= 80:
        mensagem = (
            f"### {nome_mascote} está muito feliz!\n\n"
            "Parabéns! Você arrasou na leitura! "
            "Estou muito feliz com seu desempenho!"
        )

    elif percentual <= 50:
        mensagem = (
            f"### {nome_mascote} ficou um pouquinho triste.\n\n"
            "Ainda precisamos praticar um pouquinho mais. "
            "Não desista! Eu acredito em você!"
        )

    else:
        mensagem = (
            f"### {nome_mascote} está torcendo por você!\n\n"
            "Você está indo bem! Continue praticando "
            "para melhorar ainda mais!"
        )

    caminho_imagem = os.path.join(
        PASTA_MASCOTES,
        f"{nome_mascote.lower()}.png"
    )

    return caminho_imagem, mensagem

def registrar_selecao(evt: gr.SelectData):
    return evt.index


# ----------------------------- INTERFACE: UMA ÚNICA PÁGINA -----------------------------

CAMINHO_CSS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "style.css"
)

with open(CAMINHO_CSS, "r", encoding="utf-8") as arquivo:
    CSS = arquivo.read()
    


with gr.Blocks(
    title="EcoLeitura — Alfabetização e Meio Ambiente", 
    theme=gr.themes.Soft(), css = CSS
    ) as demo:
    historico_state = gr.State([])
    mascote_state = gr.State(None)
    indice_mascote_state = gr.State(None)

    gr.Markdown(
            """
            # 🌱 EcoLeitura
            ### Aprendendo a ler, cuidando do planeta!
            Gere um texto ambiental com IA, leia em voz alta e receba um retorno automático
            para apoiar sua prática de leitura.
            """
        )

    gr.Markdown("## Escolha seu mascote!")
    gr.Markdown(
        "Escolha um amigo da natureza para acompanhar "
        "suas missões de leitura."
    )

    mascote = gr.Gallery(
        value=IMAGENS_MASCOTES,
        label="Escolha seu companheiro de aventura",
        columns=4,
        height=450,
        object_fit="contain",
        allow_preview=False,
        interactive=False
    )

    confirmar_mascote = gr.Button(
        "Escolher mascote",
        variant="primary"
    )

    mensagem_mascote = gr.Markdown()


    gr.Markdown(
        """
        ## Prepare sua aventura
        Escolha o assunto que deseja explorar e
        o nível de dificuldade da sua leitura.
        """
    )

    with gr.Group():
        with gr.Row():
            tema = gr.Dropdown(
                choices=TEMAS,
                value=TEMAS[0],
                label="Qual tema você quer explorar?"
            )

            nivel = gr.Radio(
                choices=NIVEIS,
                value="Iniciante",
                label="Escolha seu nível de leitura"
            )
       

    gr.Markdown(
        """
        ##  Sua missão de leitura
        Prepare-se para descobrir algo novo sobre a natureza!
        """
    )

    with gr.Group():
        gerar_btn = gr.Button(
            "Começar missão de leitura - Gere um texto com a IA",
            variant="primary"
        )

        status_geracao = gr.Markdown()

        texto_original = gr.Textbox(
            label="Leia o texto abaixo com atenção",
            lines=5,
            placeholder="Sua missão de leitura aparecerá aqui."
        )

    gr.Markdown("---")
    gr.Markdown(
        """
        ##  Desafio da leitura
        Chegou a hora de mostrar suas habilidades!
        Leia o texto em voz alta e grave sua leitura.
        """
    )

    with gr.Group():
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Grave sua leitura aqui"
        )

        avaliar_btn = gr.Button(
            "Concluir desafio - Vamos avaliar sua leitura?",
            variant="primary"
        )

        transcricao = gr.Textbox(
            label="O que foi reconhecido na sua leitura",
            lines=4,
            interactive=False
        )

    gr.Markdown(
        """
        ##   Suas conquistas
        Veja como foi sua aventura de leitura!
        Cada atividade é uma oportunidade de aprender.
        """
    )

    with gr.Group():
        with gr.Row(elem_classes=["area-mascote"]):
            imagem_resultado_mascote = gr.Image(
                show_label=False,
                height=180,
                interactive=False,
                visible=False,
                container=False,
                elem_classes=["imagem-mascote"]
            )

            mensagem_resultado_mascote = gr.Markdown(
                elem_classes=["balao-mascote"]
            )

        relatorio = gr.Markdown(
            "O resultado da sua missão aparecerá aqui."
        )

        gr.Markdown("### Histórico das suas aventuras")

        tabela_historico = gr.Dataframe(
            headers=["Data", "Tema", "Nível", "Resultado", "Feedback"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="Missões realizadas"
        )

        limpar_btn = gr.Button(
            "Começar uma nova aventura",
            variant="secondary"
        )

    gr.Markdown(
        """
        ---
        <small>
        AVISO: ferramenta experimental para apoio à alfabetização. O reconhecimento
        automático pode errar, especialmente com vozes infantis, sotaques e ruídos.
        A atividade deve ser acompanhada por um adulto/educador. O áudio é processado
        durante a atividade e não é salvo por este código.
        </small>
        """
    )

    gerar_btn.click(
        fn=gerar_e_atualizar,
        inputs=[tema, nivel, historico_state],
        outputs=[texto_original, status_geracao],
    )

    avaliar_btn.click(
        fn=processar_leitura,
        inputs=[
            audio,
            texto_original,
            tema,
            nivel,
            historico_state,
            mascote_state
        ],
        outputs=[
            transcricao,
            relatorio,
            tabela_historico,
            historico_state,
            imagem_resultado_mascote,
            mensagem_resultado_mascote
        ],
    )

    mascote.select(
        fn=registrar_selecao,
        outputs=[indice_mascote_state]
)

    confirmar_mascote.click(
        fn=escolher_mascote,
        inputs=[indice_mascote_state],
        outputs=[mascote_state, mensagem_mascote]
)

    limpar_btn.click(
        fn=limpar_atividade,
        inputs=[],
        outputs=[
            texto_original,
            audio,
            transcricao,
            relatorio,
            tabela_historico,
            historico_state,
            imagem_resultado_mascote,
            mensagem_resultado_mascote
        ],
    )


if __name__ == "__main__":
   demo.launch(
    allowed_paths=[
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "imagens"
        )
    ]
)