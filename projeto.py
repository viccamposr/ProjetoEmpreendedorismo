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
import re
import unicodedata
import html 
import difflib
from datetime import datetime

import gradio as gr
from faster_whisper import WhisperModel
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name(".env"))
from servicos import gerar_texto


# ----------------------------- CONFIGURAÇÕES -----------------------------

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




PASTA_MASCOTES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mascotes"
)

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

    detalhes = (
        "\n".join(f"- {linha}" for linha in linhas)
        if linhas
        else "Nenhuma diferença de palavras foi identificada."
    )

    palavras_revisao = (
    [palavra for palavra, _ in diferentes] + faltantes
    )

    return percentual, mensagem, detalhes, palavras_revisao

def colorir_transcricao(texto_original, transcricao):
    palavras_originais = normalizar(texto_original).split()
    palavras_lidas = transcricao.split()
    palavras_comparacao = normalizar(transcricao).split()

    comparador = difflib.SequenceMatcher(
        a=palavras_originais,
        b=palavras_comparacao,
        autojunk=False
    )

    cores = ["errada"] * len(palavras_lidas)

    for operacao, i1, i2, j1, j2 in comparador.get_opcodes():
        if operacao == "equal":
            for i in range(j1, j2):
                cores[i] = "correta"

    resultado = []

    for palavra, cor in zip(palavras_lidas, cores):
        palavra_segura = html.escape(palavra)

        resultado.append(
            f'<span class="palavra-{cor}">{palavra_segura}</span>'
        )

    return " ".join(resultado)


# ----------------------------- EVENTOS DA INTERFACE -----------------------------

def gerar_e_atualizar(tema, nivel, historico):
    texto = gerar_texto(tema, nivel, historico)
    return texto, "Texto gerado! Agora leia em voz alta e grave sua leitura."


def processar_leitura(audio, texto_original, tema, nivel, historico, nome_mascote):
    if not texto_original or not texto_original.strip():
        raise gr.Error("Primeiro, gere um texto ambiental.")

    transcricao = transcrever_audio(audio)
    transcricao_colorida = colorir_transcricao(
        texto_original,
        transcricao
    )

    transcricao_formatada = f"""
    <div class="resultado-transcricao">
        <strong>O que foi reconhecido na sua leitura</strong>
        <p>{transcricao_colorida}</p>
    </div>
    """
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

    relatorio = (
        "## Relatório de leitura\n\n"
        f"**Tema:** {tema}\n\n"
        f"**Nível:** {nivel}\n\n"
        f"**Correspondência estimada:** {percentual}%\n\n"
        f"**Feedback:** {mensagem}\n\n"
        "### Palavras para revisar\n\n"
        f"{detalhes}\n\n"
        "---\n\n"
        "*Esta porcentagem representa a correspondência aproximada "
        "entre as palavras do texto e a transcrição automática. "
        "Ela não comprova a pronúncia correta de cada som.*"
    )
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

    return ( transcricao_formatada, relatorio, tabela, historico,
        gr.update(
            value=imagem_mascote,
            visible=imagem_mascote is not None
        ),
        mensagem_mascote
    )


def limpar_atividade():
    return ( "", None, "", "", [], [], gr.update(value=None, visible=False), ""
    )

def escolher_mascote(selecao):
    if selecao is None:
        raise gr.Error("Escolha um mascote para continuar.")

    nome = list(MASCOTES.keys())[selecao]
    animal = MASCOTES[nome]

    mensagem = (
        f"### Olá! Eu sou {nome}!\n\n"
        f"Sou {animal.lower()} e vou acompanhar você "
        "nas suas aventuras de leitura!"
    )

    caminho_imagem = os.path.join(
        PASTA_MASCOTES,
        f"{nome.lower()}.png"
    )

    return (
        nome,
        mensagem,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=caminho_imagem, visible=True),
        gr.update(visible=True)
    )

def reescolher_mascote():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        None,
        "",
        None
    )

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

    with gr.Column(visible=True) as area_selecao_mascote:

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


    with gr.Column(visible=False) as area_mascote_escolhido:

        imagem_mascote_escolhido = gr.Image(
            show_label=False,
            height=200,
            interactive=False,
            container=False
        )

        mensagem_mascote = gr.Markdown()

        reescolher_btn = gr.Button(
            "Escolher outro mascote mascote",
            variant="secondary"
        )


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

        transcricao = gr.HTML(
            value="""
            <div class="resultado-transcricao">
                <strong>O que foi reconhecido na sua leitura</strong>
                <p>A transcrição aparecerá aqui após a avaliação.</p>
            </div>
            """
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
        outputs=[
            mascote_state,
            mensagem_mascote,
            area_selecao_mascote,
            area_mascote_escolhido,
            imagem_mascote_escolhido,
            reescolher_btn
        ]
    )

    reescolher_btn.click(
        fn=reescolher_mascote,
        inputs=[],
        outputs=[
            area_selecao_mascote,
            area_mascote_escolhido,
            imagem_mascote_escolhido,
            reescolher_btn,
            mascote_state,
            mensagem_mascote,
            indice_mascote_state
        ]
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