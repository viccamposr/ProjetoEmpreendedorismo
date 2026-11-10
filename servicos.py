import os
import re

import gradio as gr
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


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
        - Use palavras simples e adequadas ao nível.
        - Não invente informações.
        - Não use frases poéticas ou metafóricas.
        - Não use frases genéricas como "O planeta é bonito".
        - Não use comandos como "Vamos cuidar da natureza".
        - Não escreva título, perguntas ou listas.
        - Escreva somente um parágrafo.
        - Respeite a quantidade de frases definida para o nível.

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

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )

    try:
        resposta = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=256
            )
        )


        texto = (resposta.text or "").strip()
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
