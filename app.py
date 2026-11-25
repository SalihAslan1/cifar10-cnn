import argparse
from pathlib import Path

import gradio as gr
import torch

from model import MODEL_PATH, load_model, predict


def build_app(checkpoint=MODEL_PATH):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_model(checkpoint)

    def classify(image):
        if image is None:
            raise gr.Error("Bir fotoğraf yükle.")
        return predict(model, image)

    with gr.Blocks(title="CIFAR-10 Görüntü Sınıflandırma", analytics_enabled=False) as demo:
        gr.Markdown("# CIFAR-10 Görüntü Sınıflandırma")
        with gr.Row():
            image = gr.Image(label="Fotoğraf", type="pil", sources=["upload"], height=320)
            result = gr.Label(label="Model tahmini", num_top_classes=3)
        gr.Button("Sınıflandır", variant="primary").click(
            classify, inputs=image, outputs=result, api_name="classify",
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Gradio demosu")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.checkpoint).launch(server_name="127.0.0.1", server_port=args.port, share=False)
