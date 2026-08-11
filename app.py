"""
Scanner com Yolo
-----------------
Protótipo educacional de detecção de objetos em tempo real, usando
Streamlit e YOLO (Ultralytics). A captura de imagem é feita pelo
componente nativo st.camera_input, que aciona a câmera do
NAVEGADOR do usuário (client-side) — essencial para funcionar em
deploy na nuvem (Render), onde o servidor não possui câmera física.
"""

from typing import Any

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Scanner com Yolo", layout="centered")
st.title("Scanner com Yolo")
st.caption("Detecção de objetos em tempo real com YOLO + Streamlit")


# ---------------------------------------------------------------------------
# Carregamento do modelo (cacheado para não recarregar a cada interação)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo YOLO...")
def carregar_modelo(caminho_modelo: str = "yolov8n.pt") -> YOLO:
    """Carrega o modelo YOLO uma única vez e mantém em cache na sessão."""
    try:
        return YOLO(caminho_modelo)
    except Exception as erro:
        st.error(f"Falha ao carregar o modelo YOLO: {erro}")
        st.stop()


# ---------------------------------------------------------------------------
# Inferência: recebe a imagem capturada e retorna a imagem anotada + detecções
# ---------------------------------------------------------------------------
def detectar_objetos(modelo: YOLO, imagem: Image.Image) -> tuple[Any, list[str]]:
    """Executa a inferência do YOLO sobre a imagem e formata os resultados."""
    resultado = modelo.predict(imagem, verbose=False)[0]
    imagem_anotada = resultado.plot()  # array BGR com as caixas desenhadas

    deteccoes = [
        f"{modelo.names[int(caixa.cls[0])]} — {float(caixa.conf[0]):.0%} de confiança"
        for caixa in resultado.boxes
    ]
    return imagem_anotada, deteccoes


# ---------------------------------------------------------------------------
# Interface: botão que libera o widget de câmera do navegador
# ---------------------------------------------------------------------------
if "camera_ativa" not in st.session_state:
    st.session_state["camera_ativa"] = False

if st.button("Abrir câmera"):
    st.session_state["camera_ativa"] = True

if st.session_state["camera_ativa"]:
    captura = st.camera_input("Aponte a câmera para o objeto e capture")

    if captura is not None:
        try:
            imagem_original = Image.open(captura)
        except Exception as erro:
            st.error(f"Não foi possível ler a imagem capturada: {erro}")
        else:
            modelo_yolo = carregar_modelo()
            imagem_anotada, deteccoes = detectar_objetos(modelo_yolo, imagem_original)

            # Pós-processamento e visualização dos resultados
            st.image(imagem_anotada, channels="BGR", caption="Objetos detectados")

            st.subheader("Objetos encontrados")
            if deteccoes:
                for item in deteccoes:
                    st.write(f"- {item}")
            else:
                st.info("Nenhum objeto foi identificado nesta captura.")