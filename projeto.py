import gradio as gr

# --- MÓDULO 1: GERADOR DE TEXTO AMBIENTAL (LLM) ---
def gerar_texto_ambiental(tema, nivel):
    """
    Função para gerar frases focadas no Fundamental I.
    Futuramente, esta função chamará a API de um LLM (Gemini/OpenAI/Groq).
    """
    base_frases = {
        "Reciclagem": {
            "Iniciante": "O papel vai no lixo azul. O plástico vai no lixo vermelho.",
            "Intermediário": "Separar o lixo reciclável evita a poluição do solo e ajuda a salvar o meio ambiente."
        },
        "Água": {
            "Iniciante": "Feche a torneira ao escovar os dentes para economizar água.",
            "Intermediário": "A água limpa é um recurso valioso e vital para todas as plantas e animais do planeta."
        },
        "Florestas": {
            "Iniciante": "As árvores dão frutos e ajudam a limpar o ar que respiramos.",
            "Intermediário": "Proteger as florestas garante um abrigo seguro para a biodiversidade de espécies."
        }
    }
    return base_frases.get(tema, {}).get(nivel, "Devemos cuidar do nosso planeta com carinho.")

# --- MÓDULO 2: AVALIADOR DE PRONÚNCIA E LEITURA (STT / WHISPER) ---
def avaliar_pronuncia(audio_path, texto_esperado):
    """
    Função que processará o áudio gravado do microfone.
    Futuramente, integrará o modelo Whisper para transcrição e comparação.
    """
    if audio_path is None:
        return "⚠️ Nenhum áudio detectado. Por favor, grave a leitura da criança antes de avaliar."
    
    if not texto_esperado:
        return "⚠️ Por favor, gere um texto na Aba 1 antes de praticar a leitura."

    # Estrutura do relatório de resposta
    relatorio = f"""
    🟢 LEITURA PROCESSADA COM SUCESSO!
    --------------------------------------------------
    • Texto Alvo: "{texto_esperado}"
    • Status do Modelo: Áudio recebido e pronto para análise do Whisper.
    
    📊 MÉTRICAS PREVISTAS PARA A POC:
    - Taxa de Erro de Palavras (WER)
    - Precisão de Pronúncia das Palavras Ecológicas
    - Índice de Fluência Leitora
    """
    return relatorio

# --- MÓDULO 3: INTERFACE VISUAL DUAL (GRADIO BLOCKS) ---
with gr.Blocks(title="EcoLeitura - PoC IA", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 EcoLeitura: IA para Alfabetização e Meio Ambiente")
    gr.Markdown("Prova de Conceito (PoC) para a disciplina SIN 323 - Inteligência Artificial.")
    
    # --- ABA 1: GERADOR DE TEXTO ---
    with gr.Tab("📖 1. Gerar Atividade"):
        gr.Markdown("### Passo 1: Configure e gere a frase temática")
        with gr.Row():
            with gr.Column(scale=1):
                tema_input = gr.Dropdown(
                    choices=["Reciclagem", "Água", "Florestas"], 
                    label="Tema Ambiental", 
                    value="Reciclagem"
                )
                nivel_input = gr.Radio(
                    choices=["Iniciante", "Intermediário"], 
                    label="Nível de Alfabetização", 
                    value="Iniciante"
                )
                btn_gerar = gr.Button("✨ Gerar Texto com IA", variant="primary")
            
            with gr.Column(scale=2):
                texto_output = gr.Textbox(
                    label="Texto Gerado para Leitura", 
                    lines=4, 
                    interactive=False
                )
        
        btn_gerar.click(
            fn=gerar_texto_ambiental, 
            inputs=[tema_input, nivel_input], 
            outputs=texto_output
        )

    # --- ABA 2: AVALIAÇÃO DE ÁUDIO ---
    with gr.Tab("🎙️ 2. Praticar Leitura e Pronúncia"):
        gr.Markdown("### Passo 2: Grave a criança a ler o texto gerado")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="Gravador de Voz"
                )
                btn_avaliar = gr.Button("🔍 Avaliar Pronúncia", variant="primary")
            
            with gr.Column(scale=1):
                resultado_output = gr.Textbox(
                    label="Relatório de Diagnóstico da IA", 
                    lines=8, 
                    interactive=False
                )
        
        btn_avaliar.click(
            fn=avaliar_pronuncia, 
            inputs=[audio_input, texto_output], 
            outputs=resultado_output
        )

    # --- ABA 3: PAINEL DE VALIDAÇÃO (REQUISITO DA DISCIPLINA) ---
    with gr.Tab("📊 3. Validação e Métricas"):
        gr.Markdown("### Estrutura de Avaliação do Projeto")
        gr.Markdown("""
        - **Problema Real:** Dificuldade na personalização do acompanhamento de leitura e falta de engajamento em temas ambientais no Fundamental I.
        - **Métricas do Modelo:** Avaliação de acurácia utilizando a métrica *Word Error Rate* (WER).
        - **Uso Responsável:** Monitorização de vieses de sotaque infantil e garantia da privacidade dos dados de voz conforme a LGPD.
        """)

# Executa abrindo o navegador automaticamente
if __name__ == "__main__":
    demo.launch(inbrowser=True)