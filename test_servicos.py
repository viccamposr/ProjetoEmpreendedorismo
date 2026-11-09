"""Testes locais; nenhuma gravação ou chave real é enviada aos provedores."""
import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import requests
import soundfile as sf

import servicos as s


def resposta_fala():
    return {"RecognitionStatus": "Success", "NBest": [{
        "Display": "A água.", "Lexical": "a água", "PronScore": 88,
        "AccuracyScore": 86, "FluencyScore": 90, "CompletenessScore": 100,
        "Words": [{"Word": "a", "AccuracyScore": 100, "ErrorType": "None"},
                  {"Word": "água", "AccuracyScore": 65, "ErrorType": "Mispronunciation"}],
    }]}


class TestServicos(unittest.TestCase):
    def test_wer_pontuacao_e_acentos(self):
        self.assertEqual(s.calcular_wer("A água!", "a água"), 0)
        self.assertEqual(s.calcular_wer("avó", "avô"), 100)
        self.assertEqual(s.calcular_wer("água", "a água limpa"), 200)
        self.assertEqual(s.calcular_wer("a água limpa", "a limpa"), 33.3)
        self.assertEqual(s.calcular_wer("a água", ""), 100)

    def test_wer_sem_referencia(self):
        with self.assertRaises(s.ErroAplicacao):
            s.calcular_wer("...", "água")

    def test_exemplos_sem_rede(self):
        with patch.object(s.requests, "post") as post:
            for tema, niveis in s.EXEMPLOS.items():
                for nivel in niveis:
                    texto, origem = s.gerar_texto_ambiental(tema, nivel)
                    self.assertTrue(texto)
                    self.assertIn("sem geração por IA", origem)
            post.assert_not_called()

    def test_geracao_real_e_limite(self):
        resposta = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "A água faz a planta crescer. Cada gota conta."}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "teste"}), patch.object(s, "requisitar_json", return_value=resposta):
            texto, origem = s.gerar_texto_ambiental("Água", "Iniciante", "Gerar com IA")
            self.assertIn("Cada gota", texto)
            self.assertIn("gerado por IA", origem)
            resposta["candidates"][0]["finishReason"] = "MAX_TOKENS"
            with self.assertRaises(s.ErroAplicacao):
                s.gerar_texto_ambiental("Água", "Iniciante", "Gerar com IA")

    def test_modelo_indisponivel_explica_como_corrigir(self):
        resposta = Mock(status_code=404)
        resposta.raise_for_status.side_effect = requests.HTTPError("segredo", response=resposta)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "teste"}), patch.object(s.requests, "post", return_value=resposta) as post:
            with self.assertRaisesRegex(s.ErroAplicacao, "GEMINI_MODEL") as erro:
                s.gerar_texto_ambiental("Água", "Iniciante", "Gerar com IA")
            self.assertNotIn("segredo", str(erro.exception))
            self.assertNotIn("mensagem_404", post.call_args.kwargs)

    def test_atividade_longa_preserva_frase_completa(self):
        primeira = "Jogamos o lixo na lixeira certa para cuidar da nossa linda natureza verde."
        resposta = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": primeira + " O papel velho vira um caderno novo e ajuda o planeta."}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "teste"}), patch.object(s, "requisitar_json", return_value=resposta):
            texto, _ = s.gerar_texto_ambiental("Reciclagem", "Iniciante", "Gerar com IA")
            self.assertEqual(texto, primeira)
            self.assertLessEqual(len(s.normalizar(texto)), 16)

    def test_frase_longa_nao_e_cortada_no_meio(self):
        resposta = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "palavra " * 30 + "."}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "teste"}), patch.object(s, "requisitar_json", return_value=resposta):
            with self.assertRaises(s.ErroAplicacao):
                s.gerar_texto_ambiental("Reciclagem", "Iniciante", "Gerar com IA")

    def test_modelo_em_branco_usa_padrao(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "teste", "GEMINI_MODEL": " "}), patch.object(s, "requisitar_json", side_effect=s.ErroAplicacao("interrompido")) as post:
            with self.assertRaises(s.ErroAplicacao):
                s.gerar_texto_ambiental("Água", "Iniciante", "Gerar com IA")
            self.assertIn('/' + s.MODELO_GEMINI_PADRAO + ':generateContent', post.call_args.args[0])

    def test_sem_chaves_nao_simula_avaliacao(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(s.requests, "post") as post:
            with self.assertRaisesRegex(s.ErroAplicacao, "Configure"):
                s.avaliar_pronuncia("teste.wav", "A água.", True)
            post.assert_not_called()

    def test_sem_autorizacao_nao_envia(self):
        with patch.object(s.requests, "post") as post:
            with self.assertRaisesRegex(s.ErroAplicacao, "Autorize"):
                s.avaliar_pronuncia("teste.wav", "A água.", False)
            post.assert_not_called()

    def test_notas_vem_do_servico(self):
        resultado = s.montar_resultado(resposta_fala(), "A água.", 2)
        self.assertEqual(resultado["dados"]["notas"]["geral"], 88)
        self.assertEqual(resultado["dados"]["wer_percentual"], 0)
        self.assertIn("água", resultado["relatorio"])
        self.assertEqual(resultado["palavras"][1][1], 65)

    def test_omissao_sem_nota_inventada(self):
        dados = resposta_fala()
        dados["NBest"][0]["Words"].append({"Word": "limpa", "ErrorType": "Omission"})
        self.assertIsNone(s.montar_resultado(dados, "A água limpa.", 2)["palavras"][-1][1])

    def test_fala_ausente_ou_resposta_incompleta(self):
        for dados in [{"RecognitionStatus": "NoMatch"}, {"RecognitionStatus": "Success", "NBest": []}]:
            with self.assertRaises(s.ErroAplicacao):
                s.montar_resultado(dados, "A água.", 2)
        for valor in [None, -1, 101, float("nan"), True]:
            dados = resposta_fala()
            dados["NBest"][0]["PronScore"] = valor
            with self.assertRaises(s.ErroAplicacao):
                s.montar_resultado(dados, "A água.", 2)

    def test_audio_convertido_para_pcm_mono_16k(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "audio.wav"
            onda = 0.1 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
            sf.write(caminho, np.column_stack([onda, onda]), 48000)
            wav, duracao = s.preparar_audio(caminho)
            with sf.SoundFile(io.BytesIO(wav)) as arquivo:
                self.assertEqual((arquivo.channels, arquivo.samplerate, arquivo.subtype), (1, 16000, "PCM_16"))
            self.assertEqual(duracao, 1)
            sf.write(caminho, np.zeros(16000), 16000)
            with self.assertRaisesRegex(s.ErroAplicacao, "baixo"):
                s.preparar_audio(caminho)
            sf.write(caminho, np.zeros(31 * 16000), 16000)
            with self.assertRaisesRegex(s.ErroAplicacao, "30 segundos"):
                s.preparar_audio(caminho)

    def test_envio_referencia_e_idioma(self):
        with patch.dict(os.environ, {"AZURE_SPEECH_KEY": "teste", "AZURE_SPEECH_REGION": "brazilsouth"}), patch.object(s, "preparar_audio", return_value=(b"wav", 2)), patch.object(s, "requisitar_json", return_value=resposta_fala()) as post:
            s.avaliar_pronuncia("teste.wav", "A água.", True)
            envio = post.call_args.kwargs
            config = json.loads(base64.b64decode(envio["headers"]["Pronunciation-Assessment"]))
            self.assertEqual(config["ReferenceText"], "A água.")
            self.assertTrue(config["EnableMiscue"])
            self.assertEqual(envio["params"]["language"], "pt-BR")

    def test_erros_http_nao_expoem_detalhes(self):
        for status in [401, 403, 429, 500]:
            resposta = Mock(status_code=status)
            resposta.raise_for_status.side_effect = requests.HTTPError("segredo", response=resposta)
            with patch.object(s.requests, "post", return_value=resposta):
                with self.assertRaises(s.ErroAplicacao) as erro:
                    s.requisitar_json("https://example.com")
                self.assertNotIn("segredo", str(erro.exception))
        with patch.object(s.requests, "post", side_effect=requests.Timeout):
            with self.assertRaises(s.ErroAplicacao):
                s.requisitar_json("https://example.com")


if __name__ == "__main__":
    unittest.main()
