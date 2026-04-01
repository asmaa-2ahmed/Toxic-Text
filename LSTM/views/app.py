import gradio as gr
from src.image_to_text import caption_image
from src.inference import predict_text


def analyze(image, query: str):
   
    if image is None:
        return (
            "",
            {},
            "⚠️ Please upload an image first.",
        )

    caption = caption_image(image)

    combined = f"{query.strip()} {caption}".strip() if query.strip() else caption

    if not combined:
        return caption, {}, "⚠️ Could not generate a meaningful caption."

    label, confidence, all_scores = predict_text(combined)

    confidence_dict = all_scores

    result_text = f"Confidence: {confidence:.2%}"

    return caption, confidence_dict, result_text


# Gradio interface

with gr.Blocks( title="Toxic Text Classifier" ) as demo:

    gr.Markdown("# 🛑 Toxic Text Classifier", elem_id="title")
    gr.Markdown(
        "Upload an image — the app will caption it automatically, "
        "combine it with your optional query, then classify the content.",
        elem_id="subtitle",
    )

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Upload Image",
                type="pil",
                height=280,
            )
            query_input = gr.Textbox(
                label="💬 Additional Query (optional)",
                placeholder="e.g. Is this content harmful?",
                lines=3,
            )
            analyze_btn = gr.Button(
                "🔍 Analyze",
                variant="primary",
                elem_id="analyze-btn",
            )

        # ── Right column: outputs ─────────────────────────────────────────
        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="📝 Auto-Generated Image Caption",
                interactive=False,
                lines=3,
            )
            label_output = gr.Label(
                label="🏷️ Predicted Category",
                num_top_classes=3,
            )
            confidence_output = gr.Textbox(
                label="📊 Confidence",
                interactive=False,
            )

    # ── Wire up ───────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze,
        inputs=[image_input, query_input],
        outputs=[caption_output, label_output, confidence_output],
    )


def launch():
    demo.launch(
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    launch()