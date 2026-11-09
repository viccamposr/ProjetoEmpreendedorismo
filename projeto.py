"""Interface web do EcoLeitura. Execute: python projeto.py."""
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from servicos import ErroAplicacao, configuracao, gerar_texto_ambiental, avaliar_pronuncia

load_dotenv(Path(__file__).with_name(".env"))


def gerar(tema, nivel, modo):
    try:
        texto, origem = gerar_texto_ambiental(tema, nivel, modo)
        return texto, origem, None, "", [], None
    except ErroAplicacao as exc:
        # Falhas do provedor ficam no status, preservando a atividade anterior.
        return gr.skip(), f"Não foi possível criar a atividade: {exc}", gr.skip(), gr.skip(), gr.skip(), gr.skip()


def avaliar(audio, texto, autorizado):
    try:
        resultado = avaliar_pronuncia(audio, texto, autorizado)
        return resultado["relatorio"], resultado["palavras"], resultado["dados"]
    except ErroAplicacao as exc:
        return f"Avaliação não realizada: {exc}", [], None


def criar_interface():
    cfg = configuracao()
    with gr.Blocks(title="EcoLeitura", delete_cache=(300, 3600)) as demo:
        gr.Markdown("# 🌿 EcoLeitura\nLeia, descubra e cuide do planeta.")
        gr.Markdown("Escolha uma atividade, grave uma leitura de **até 30 segundos** e veja quais palavras pode praticar. Português do Brasil.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Prepare a leitura")
                tema = gr.Dropdown(["Reciclagem", "Água", "Florestas"], value="Reciclagem", label="Tema")
                nivel = gr.Radio(["Iniciante", "Intermediário"], value="Iniciante", label="Nível")
                modo = gr.Radio(["Exemplos locais", "Gerar com IA"], value="Gerar com IA" if cfg["geracao"] else "Exemplos locais", label="Origem do texto")
                btn_gerar = gr.Button("Criar atividade", variant="primary")
                origem = gr.Textbox(label="Status da atividade", interactive=False)
            with gr.Column(scale=2):
                texto = gr.Textbox(label="Texto para ler", lines=4, interactive=False)
                gr.Markdown("## 2. Leia em voz alta")
                audio = gr.Audio(sources=["microphone", "upload"], type="filepath", format="wav", label="Sua leitura")
                autorizado = gr.Checkbox(label="Autorizo enviar esta gravação e o texto ao Azure Speech para avaliação. Para crianças, a atividade deve ser acompanhada pelo responsável.", value=False)
                btn_avaliar = gr.Button("Avaliar minha leitura", variant="primary")
        gr.Markdown("## 3. Seu relatório")
        relatorio = gr.Textbox(label="Resultado e sugestões", lines=14, interactive=False)
        palavras = gr.Dataframe(headers=["Palavra", "Precisão (0–100)", "Observação"], datatype=["str", "number", "str"], interactive=False, label="Avaliação por palavra")
        with gr.Accordion("Dados da avaliação", open=False):
            dados = gr.JSON(label="Relatório estruturado")
        with gr.Accordion("Como interpretar e configurar", open=False):
            gr.Markdown(
                f"**Geração por IA:** {'configurada' if cfg['geracao'] else 'configure GEMINI_API_KEY no .env'}.\n\n"
                f"**Avaliação acústica:** {'configurada' if cfg['fala'] else 'configure AZURE_SPEECH_KEY e AZURE_SPEECH_REGION no .env'}.\n\n"
                "As notas são estimativas do modelo. Ruído, microfone e sotaque podem alterar os resultados. "
                "A WER compara palavras reconhecidas e esperadas; não mede pronúncia. Revise com um educador. "
                "Sem credenciais não são inventadas notas.\n\n"
                "O tema e o nível são enviados ao Gemini apenas no modo IA. Ao avaliar, a gravação e o texto "
                "são enviados ao Azure. Arquivos temporários da interface são limpos periodicamente após uma hora; "
                "a aplicação não mantém histórico próprio. Consulte as políticas dos provedores antes de usar com dados reais."
            )
        btn_gerar.click(gerar, [tema, nivel, modo], [texto, origem, audio, relatorio, palavras, dados], api_visibility="private")
        btn_avaliar.click(avaliar, [audio, texto, autorizado], [relatorio, palavras, dados], api_visibility="private")
        # Só alterações feitas pelo usuário limpam o relatório; não disputar com a resposta da avaliação.
        audio.input(lambda: ("", [], None), outputs=[relatorio, palavras, dados], api_visibility="private")
    return demo


if __name__ == "__main__":
    criar_interface().queue(default_concurrency_limit=2, max_size=16).launch(
        inbrowser=True, server_name="127.0.0.1", share=False,
        theme=gr.themes.Soft(primary_hue="emerald"), max_file_size="15mb",
    )
