import os
import gradio as gr
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List, Dict
import glob
from collections import defaultdict

from transformers import (AutoImageProcessor,
                          ResNetForImageClassification)

from src.labelmap import DR_LABELMAP
from src.fetch_files import fetch_files


class App:
    """ Demonstration of the Diabetic Retinopathy model as a Gradio app. """

    def __init__(self) -> None:
        """ Constructor. """

        ckpt_name = "2023-12-24_20-02-18_30345221_V100_x4_resnet34/"

        path = f"release_ckpts/{ckpt_name}/inference/"

        if not os.path.exists(path):
            raise Exception(f"Checkpoint not found at {path}")

        self.image_processor = AutoImageProcessor.from_pretrained(path)

        self.model = ResNetForImageClassification.from_pretrained(path)

        example_lists = self._load_example_lists()

        device = 'GPU' if torch.cuda.is_available() else 'CPU'

        css = ".output-image, .input-image, .image-preview {height: 600px !important}"

        with gr.Blocks(css=css) as ui:
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        predict_btn = gr.Button("Predict", size="lg")
                    with gr.Row():
                        gr.Markdown(f"Running on {device}")
                with gr.Column(scale=4):
                    # output = gr.Textbox(label="Retinopathy level prediction")
                    output = gr.Label(num_top_classes=len(DR_LABELMAP),
                                      label="Retinopathy level prediction")
                with gr.Column(scale=4):
                    gr.Markdown("![](https://raw.githubusercontent.com/SDAIA-KAUST-AI/diabetic-retinopathy-detection/main/media/logo1.png)")
            with gr.Row():
                with gr.Column(scale=9, min_width=100):
                    image = gr.Image(label="Retina scan")
                with gr.Column(scale=1, min_width=150):
                    for cls_id in range(len(example_lists)):
                        label = DR_LABELMAP[cls_id]
                        with gr.Tab(f"{cls_id} : {label}"):
                            gr.Examples(
                                example_lists[cls_id],
                                inputs=[image],
                                outputs=[output],
                                fn=self.predict,
                                examples_per_page=10,
                                run_on_click=True)

            predict_btn.click(
                fn=self.predict,
                inputs=image,
                outputs=output,
                api_name="predict")
        
        self.ui = ui

    def launch(self) -> None:
        """ Launch the application, blocking. """
        if 'LABEEB' in os.environ:
            kwargs = dict(share=False, debug=True,
                          server_port=8050, server_name="0.0.0.0")
            if 'ROUTE' in os.environ:
                route = os.environ["ROUTE"]
                kwargs['root_path'] = route
        else:
            kwargs = dict(share=True)
        self.ui.queue().launch(**kwargs)

    def predict(self, image: Optional[np.ndarray]) -> Dict[str, float]:
        """ Gradio callback for pricessing of an image.

        Args:
            image (Optional[np.ndarray]): Provided image.

        Returns:
            Dict[str, float]: Label-compatible dict.
        """

        if image is None:
            return dict()
        cls_name, prob, probs = self._infer(image)
        message = f"Predicted class={cls_name}, prob={prob:.3f}"
        print(message)
        probs_dict = {f"{i} - {DR_LABELMAP[i]}": float(v)
                      for i, v in enumerate(probs)}
        return probs_dict
    
    def _infer(self, image_chw: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """ Low-level method to perform neural network inference.

        Args:
            image_chw (np.ndarray): Provided image.

        Returns:
            Tuple[str, float, np.ndarray]:
                - Most probable class name
                - Probability of the most probable class name.
                - Probablilities of all classes in the order of
                  being listed in the label map.
        """

        assert isinstance(self.model, ResNetForImageClassification)

        inputs = self.image_processor(image_chw, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs)

        logits_batch = output.logits
        assert len(logits_batch.shape) == 2
        assert logits_batch.shape[0] == 1
        logits = logits_batch[0]
        probs = torch.softmax(logits, dim=-1)
        predicted_label = int(probs.argmax(-1).item())
        prob = probs[predicted_label].item()
        cls_name = self.model.config.id2label[predicted_label]
        return cls_name, prob, probs.numpy()

    @staticmethod
    def _load_example_lists() -> Dict[int, List[str]]:
        """ Load example retina images from disk.

        Returns:
            Dict[int, List[str]]: Dictionary of cls_id -> list of images paths.
        """

        example_flat_list = glob.glob("demo_data/demo/**/*.jpeg")

        example_lists: Dict[int, List[str]] = defaultdict(list)
        for path in example_flat_list:
            dir, _ = os.path.split(path)
            _, subdir = os.path.split(dir)
            try:
                cls_id = int(subdir)
            except ValueError:
                print(f"Cannot parse path {path}")
                continue
            example_lists[cls_id].append(path)
        return example_lists


def main():
    """ App entry point. """
    fetch_files()
    app = App()
    app.launch()


if __name__ == "__main__":
    main()
