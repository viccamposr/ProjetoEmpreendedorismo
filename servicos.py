"""Geração, validação de áudio e avaliação acústica, independentes da interface."""
import base64
import io
import json
import math
import os
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly


class ErroAplicacao(Exception):
    """Mensagem segura para exibir na interface, sem chaves ou resposta bruta."""


EXEMPLOS = {
    "Reciclagem": {
        "Iniciante": ["O papel vai no lixo azul. Vamos cuidar da nossa escola.", "A lata pode virar uma lata nova. Separe o lixo com cuidado."],
        "Intermediário": ["Separar o lixo reciclável ajuda a cuidar do solo. Cada pessoa pode fazer sua parte em casa e na escola.", "Uma garrafa usada pode ganhar uma nova vida. Separar os materiais facilita o trabalho de quem cuida da reciclagem."],
    },
    "Água": {
        "Iniciante": ["Feche a torneira ao escovar os dentes. Cada gota conta.", "A água faz a planta crescer. Vamos usar só o necessário."],
        "Intermediário": ["A água limpa é importante para as plantas e os animais. Podemos evitar o desperdício com pequenas atitudes todos os dias.", "Ao notar uma torneira pingando, avise um adulto. Cuidar da água ajuda a proteger a vida no nosso planeta."],
    },
    "Florestas": {
        "Iniciante": ["A árvore dá sombra e frutos. O pássaro faz seu ninho nela.", "Na mata vivem muitos animais. Vamos cuidar das árvores."],
        "Intermediário": ["As florestas abrigam muitos animais e plantas. Proteger esses espaços ajuda a manter a água e o ar mais limpos.", "Uma árvore pode ser a casa de muitos seres vivos. Ao proteger a floresta, ajudamos a preservar essa grande diversidade."],
    },
}


def configuracao():
    return {"geracao": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "fala": bool(os.getenv("AZURE_SPEECH_KEY", "").strip() and os.getenv("AZURE_SPEECH_REGION", "").strip())}


MODELO_GEMINI_PADRAO = "gemini-3.5-flash-lite"


def requisitar_json(url, *, mensagem_404=None, **kwargs):
    try:
        resposta = requests.post(url, timeout=(10, 60), **kwargs)
        resposta.raise_for_status()
        payload = resposta.json()
        if not isinstance(payload, dict):
            raise ValueError("Resposta inesperada")
        return payload
    except requests.Timeout:
        raise ErroAplicacao("O serviço demorou a responder. Tente novamente.") from None
    except requests.HTTPError as exc:
        status = exc.response.status_code
        mensagem = {401: "Credencial recusada. Confira o arquivo .env.",
                    404: mensagem_404 or "Recurso não encontrado no serviço (HTTP 404). Confira a configuração.",
                    403: "Acesso recusado. Confira as permissões e a região do serviço.",
                    429: "Limite de uso atingido. Aguarde e confira a cota do serviço."}.get(
                        status, f"O serviço não concluiu a solicitação (HTTP {status}). Tente novamente.")
        raise ErroAplicacao(mensagem) from None
    except requests.RequestException:
        raise ErroAplicacao("Não foi possível conectar ao serviço. Verifique sua conexão.") from None
    except ValueError:
        raise ErroAplicacao("O serviço retornou uma resposta inválida. Tente novamente.") from None


def gerar_texto_ambiental(tema, nivel, modo="Exemplos locais"):
    if tema not in EXEMPLOS or nivel not in EXEMPLOS[tema]:
        raise ErroAplicacao("Escolha um tema e um nível válidos.")
    if modo == "Exemplos locais":
        return random.choice(EXEMPLOS[tema][nivel]), "Exemplo local: texto pré-escrito, sem geração por IA."
    if modo != "Gerar com IA":
        raise ErroAplicacao("Escolha uma origem válida para o texto.")
    chave = os.getenv("GEMINI_API_KEY", "").strip()
    if not chave:
        raise ErroAplicacao("Configure GEMINI_API_KEY no .env ou selecione Exemplos locais.")
    modelo = os.getenv("GEMINI_MODEL", MODELO_GEMINI_PADRAO).strip() or MODELO_GEMINI_PADRAO
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", modelo):
        raise ErroAplicacao("GEMINI_MODEL inválido. Use apenas o identificador do modelo.")
    limite = 16 if nivel == "Iniciante" else 28
    prompt = (
        f"Crie uma atividade de leitura sobre {tema}, em português brasileiro, para o Fundamental I. "
        f"Nível {nivel}. Use uma ou duas frases, de 8 a {limite} palavras no total, vocabulário simples, "
        "conteúdo educativo e ambiental correto. Não inclua título, listas, números, markdown "
        "ou explicações. Retorne somente o texto a ser lido."
    )
    dados = requisitar_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent",
        mensagem_404=("O modelo Gemini configurado não está disponível para esta chave (HTTP 404). "
                      "Atualize GEMINI_MODEL no .env para um modelo com acesso à geração de texto "
                      "e reinicie o aplicativo. Para continuar sem a API, selecione Exemplos locais."),
        headers={"x-goog-api-key": chave},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.8, "maxOutputTokens": 2048}},
    )
    try:
        candidato = dados["candidates"][0]
        if candidato.get("finishReason") != "STOP":
            raise ValueError("Geração incompleta")
        texto = " ".join(p["text"] for p in candidato["content"]["parts"] if "text" in p and not p.get("thought")).strip()
        if len(normalizar(texto)) > limite:
            # Modelos podem ignorar o tamanho solicitado. Aproveite apenas frases completas
            # dentro do limite, sem cortar uma frase ou palavra no meio.
            frases = re.split(r"(?<=[.!?])\s+", texto)
            selecionadas = []
            for frase in frases:
                if len(normalizar(" ".join([*selecionadas, frase]))) > limite:
                    break
                selecionadas.append(frase)
            texto = " ".join(selecionadas)
        if not 8 <= len(normalizar(texto)) <= limite or len(texto) > 400:
            raise ValueError("Tamanho inesperado")
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise ErroAplicacao("A IA não retornou uma atividade curta válida. Tente gerar novamente.") from None
    return texto, f"Texto gerado por IA ({modelo}). Revise a atividade antes da leitura."


def normalizar(texto):
    # Preserva acentos: 'avó' e 'avô' não são a mesma palavra.
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", texto).casefold(), re.UNICODE)


def calcular_wer(referencia, transcricao):
    """Distância de Levenshtein por palavras. WER pode superar 100%."""
    esperado, falado = normalizar(referencia), normalizar(transcricao)
    if not esperado:
        raise ErroAplicacao("O texto de referência está vazio.")
    anterior = list(range(len(falado) + 1))
    for i, palavra in enumerate(esperado, 1):
        atual = [i]
        for j, outra in enumerate(falado, 1):
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + (palavra != outra)))
        anterior = atual
    return round(anterior[-1] / len(esperado) * 100, 1)


def preparar_audio(caminho):
    try:
        arquivo = Path(caminho)
        if not arquivo.is_file() or arquivo.stat().st_size > 15 * 1024 * 1024:
            raise ErroAplicacao("Envie um arquivo de áudio válido de até 15 MB.")
        with sf.SoundFile(arquivo) as fonte:
            duracao = len(fonte) / fonte.samplerate
            if not 0.3 <= duracao <= 30:
                raise ErroAplicacao("Grave entre 0,3 e 30 segundos. Use uma frase por tentativa se necessário.")
            if fonte.channels > 2 or not 8000 <= fonte.samplerate <= 192000:
                raise ErroAplicacao("Use áudio mono ou estéreo entre 8 e 192 kHz.")
            taxa = fonte.samplerate
            sinal = fonte.read(dtype="float32", always_2d=True).mean(axis=1)
    except (OSError, RuntimeError, ValueError, TypeError):
        raise ErroAplicacao("Não foi possível abrir o áudio. Grave novamente ou envie um WAV válido.") from None
    if not np.isfinite(sinal).all() or np.max(np.abs(sinal)) < 0.001:
        raise ErroAplicacao("O áudio está vazio ou muito baixo. Aproxime-se do microfone e grave novamente.")
    divisor = math.gcd(taxa, 16000)
    if taxa != 16000:
        sinal = resample_poly(sinal, 16000 // divisor, taxa // divisor)
    memoria = io.BytesIO()
    sf.write(memoria, np.clip(sinal, -1, 1), 16000, format="WAV", subtype="PCM_16")
    return memoria.getvalue(), duracao


def nota(dados, campo):
    valor = dados.get(campo)
    if isinstance(valor, bool) or not isinstance(valor, (float, int)) or not math.isfinite(valor) or not 0 <= valor <= 100:
        raise ErroAplicacao("O serviço não retornou todas as notas esperadas. Nenhuma nota foi estimada localmente.")
    return round(valor, 1)


def montar_resultado(dados, texto, duracao):
    if dados.get("RecognitionStatus") != "Success":
        raise ErroAplicacao("Não foi possível reconhecer a fala. Grave novamente em um ambiente silencioso.")
    try:
        melhor = dados["NBest"][0]
        avaliacao = melhor.get("PronunciationAssessment", melhor)
        notas = {nome: nota(avaliacao, campo) for nome, campo in {
            "geral": "PronScore", "precisao": "AccuracyScore", "fluencia": "FluencyScore", "completude": "CompletenessScore"}.items()}
        transcricao = melhor.get("Display") or dados.get("DisplayText")
        lexical = melhor.get("Lexical")
        if not isinstance(transcricao, str) or not isinstance(lexical, str) or not lexical.strip():
            raise ValueError("Sem transcrição")
        palavras = []
        nomes = {"None": "Sem desvio sinalizado", "Mispronunciation": "Praticar pronúncia",
                 "Omission": "Palavra não identificada", "Insertion": "Palavra acrescentada"}
        for item in melhor["Words"]:
            detalhe = item.get("PronunciationAssessment", item)
            tipo = detalhe.get("ErrorType", "None")
            valor = None if tipo == "Omission" else nota(detalhe, "AccuracyScore")
            palavras.append([str(item["Word"]), valor, nomes.get(tipo, "Revisar com educador")])
        if not palavras:
            raise ValueError("Sem palavras")
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise ErroAplicacao("A resposta da avaliação está incompleta. Tente novamente.") from None
    wer = calcular_wer(texto, lexical)
    praticar = list(dict.fromkeys(p[0] for p in palavras if p[2] != nomes["None"]))
    sugestao = ("Pratique devagar estas palavras: " + ", ".join(praticar[:8]) + ". Depois leia a frase inteira."
                if praticar else "Nenhum desvio foi sinalizado por palavra. Continue praticando a leitura com calma.")
    relatorio = (
        f"AVALIAÇÃO DA LEITURA · português brasileiro\n\n"
        f"Nota geral do modelo: {notas['geral']}/100\n"
        f"Precisão dos sons: {notas['precisao']}/100\n"
        f"Fluência: {notas['fluencia']}/100\n"
        f"Completude da leitura: {notas['completude']}/100\n\n"
        f"Texto esperado: {texto}\nTexto reconhecido: {transcricao}\n"
        f"WER (diferença de palavras): {wer}% — não é uma nota de pronúncia.\n\n"
        f"{sugestao}\n\nAs notas são estimativas automáticas. Ruído, sotaque e falhas de reconhecimento "
        "podem afetar o resultado; revise as palavras sinalizadas com um educador."
    )
    return {"relatorio": relatorio, "palavras": palavras, "dados": {
        "provedor": "Azure Speech Pronunciation Assessment", "idioma": "pt-BR",
        "texto_esperado": texto, "transcricao": transcricao, "duracao_segundos": round(duracao, 2),
        "notas": notas, "wer_percentual": wer,
        "palavras": [{"palavra": p[0], "precisao": p[1], "observacao": p[2]} for p in palavras],
    }}


def avaliar_pronuncia(audio_path, texto_esperado, autorizado=False):
    if not audio_path:
        raise ErroAplicacao("Grave ou envie sua leitura antes de avaliar.")
    if not isinstance(texto_esperado, str) or not normalizar(texto_esperado):
        raise ErroAplicacao("Crie uma atividade antes de avaliar.")
    if len(texto_esperado) > 500 or len(normalizar(texto_esperado)) > 40:
        raise ErroAplicacao("Use um texto curto de até 40 palavras e 500 caracteres.")
    if autorizado is not True:
        raise ErroAplicacao("Autorize o envio do áudio e do texto para realizar a avaliação.")
    chave = os.getenv("AZURE_SPEECH_KEY", "").strip()
    regiao = os.getenv("AZURE_SPEECH_REGION", "").strip()
    if not chave or not regiao:
        raise ErroAplicacao("Configure AZURE_SPEECH_KEY e AZURE_SPEECH_REGION no .env. A avaliação precisa de um serviço acústico.")
    if not re.fullmatch(r"[a-z0-9-]+", regiao):
        raise ErroAplicacao("AZURE_SPEECH_REGION inválida. Use o identificador da região do recurso.")
    wav, duracao = preparar_audio(audio_path)
    parametros = {"ReferenceText": texto_esperado, "GradingSystem": "HundredMark",
                  "Granularity": "Phoneme", "Dimension": "Comprehensive", "EnableMiscue": True}
    dados = requisitar_json(
        f"https://{regiao}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1",
        params={"language": "pt-BR", "format": "detailed"}, data=wav,
        headers={"Ocp-Apim-Subscription-Key": chave, "Accept": "application/json",
                 "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                 "Pronunciation-Assessment": base64.b64encode(json.dumps(parametros, ensure_ascii=False).encode("utf-8")).decode("ascii")},
    )
    return montar_resultado(dados, texto_esperado, duracao)
