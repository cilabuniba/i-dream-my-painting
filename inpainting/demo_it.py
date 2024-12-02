import os
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
import torch
from models.image_to_text import get_pretrained_prompt_generator, output_to_color_dict
from models.text_to_image import (
    get_freestyle_layout_mask,
    get_freestyle_text_inputs,
    get_pretrained_mm_inpainting_pipe,
)
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose, Resize
from torchvision.transforms.functional import pil_to_tensor
from transformers import pipeline
import random


# MODELS
@st.cache_resource
def init_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = (
        Path("models") / "llava" / "multimask" / "checkpoint-2884" / "adapter"
    )
    prompt_generator_processor, prompt_generator = get_pretrained_prompt_generator(
        adapter_path
    )
    sd2_path = Path("models") / "stable-diffusion-2-inpainting"
    adapter_path = (
        Path("models")
        / "sd"
        / "rca"
        / "checkpoint-2884"
        / "pytorch_lora_weights.safetensors"
    )
    mm_inpainting_pipe = get_pretrained_mm_inpainting_pipe(
        sd2_path, adapter_path, device
    )

    # translate
    # Use a pipeline as a high-level helper
    tr_enit_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-it")
    tr_iten_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-it-en")
    return (
        prompt_generator_processor,
        prompt_generator,
        mm_inpainting_pipe,
        tr_enit_pipe,
        tr_iten_pipe,
        device,
    )


prompt_generator_processor, prompt_generator, mm_inpainting_pipe, tr_enit_pipe, tr_iten_pipe, device = init_models()


def generate_prompts(masked_image, color_names):
    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDiscover and describe the objects hidden behind the masks of the colors: {', '.join(color_names)}. ASSISTANT:"
    inputs = prompt_generator_processor(prompt, masked_image, return_tensors="pt").to(
        device
    )

    # autoregressively complete prompt
    output = prompt_generator.generate(
        **inputs, max_new_tokens=512, do_sample=True, temperature=temperature
    )
    output = prompt_generator_processor.decode(output[0], skip_special_tokens=True)
    color_output = output_to_color_dict(output, color_names)
    return color_output


# TITLE
st.markdown("# 🎨 I Dream My Painting: Demo")

# DRAWING OPTIONS
# Specify canvas parameters in application
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
# )
# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# if drawing_mode == "point":
#     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
colors_map = {
    "🟥 Rosso": "#ff0000",
    "🟦 Blu": "#0000ff",
    "🟨 Giallo": "#ffff00",
    "🟩 Verde": "#00ff00",
    "🧊 Ciano": "#00ffff",
}
inv_colors_map = {
    "#ff0000": "red",
    "#0000ff": "blue",
    "#ffff00": "yellow",
    "#00ff00": "green",
    "#00ffff": "cyan",
}
colors_str_map = {
    "red": "🟥",
    "blue": "🟦",
    "yellow": "🟨",
    "green": "🟩",
    "cyan": "🧊",
}

stroke_color = st.sidebar.radio(
    "Seleziona un colore:",
    list(colors_map.keys()),
)
temperature = st.sidebar.slider("Temperatura:", 0., 2.0, 0.5, 0.1)
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Immagine:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
transforms = Compose(
    [Resize(512, interpolation=transforms.InterpolationMode.BILINEAR), CenterCrop(512)]
)

# use PIL to open bg_image
if bg_image is not None:
    bg_image = Image.open(bg_image)
    if not bg_image.mode == "RGB":
        bg_image = bg_image.convert("RGB")
    # center crop to 512x512
    bg_image = transforms(bg_image)
    st.session_state["bg_image"] = bg_image

if st.sidebar.button("Immagine casuale"):
    # open random image from data/mm_inp_dataset/images/test
    random_image = random.choice(os.listdir("data/mm_inp_dataset/images/test"))
    bg_image = Image.open(f"data/mm_inp_dataset/images/test/{random_image}")
    if not bg_image.mode == "RGB":
        bg_image = bg_image.convert("RGB")
    # center crop to 512x512
    bg_image = transforms(bg_image)
    st.session_state["bg_image"] = bg_image

# LAYOUT
r1_col1, r1_col2 = st.columns(2)

# DRAWING CANVAS
with r1_col1:
    st.markdown("## Input")
    st.info("**Disegna sull'immagine** per selezionare le regioni da completare.", icon="🖌️")
    canvas_result = None
    if "bg_image" in st.session_state:
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color=colors_map[stroke_color],
            # stroke_width=stroke_width,
            stroke_width=1,
            stroke_color=colors_map[stroke_color],
            # background_color=bg_color,
            background_image=st.session_state["bg_image"],
            update_streamlit=realtime_update,
            height=512,
            width=512,
            drawing_mode="rect",
            # point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            point_display_radius=0,
            key="canvas",
        )
    else:
        # Ask user to upload an image
        st.warning("Perfavore **carica l'immagine** di un dipinto.", icon="🖼️")
        canvas_result = None
        if "prompts" in st.session_state:
            del st.session_state["prompts"]
        if "disable_inpaint" in st.session_state:
            st.session_state["disable_inpaint"] = True

with r1_col2:
    st.markdown("## Risultato")
    st.info("Crea i prompt e clicca **Inpaint!** per vedere la magia!.", icon="🪄")

r2_col1, r2_col2 = st.columns(2)

if not "disable_inpaint" in st.session_state:
    st.session_state["disable_inpaint"] = True
with r2_col1:
    if st.button("Crea i prompt", type="primary"):
        # ON UPDATE
        # Do something interesting with the image data and paths
        if (
            canvas_result is not None
            and canvas_result.image_data is not None
            and canvas_result.json_data is not None
        ):
            # image data
            colored_masks = canvas_result.image_data
            overlay_masks = Image.fromarray(colored_masks, "RGBA")
            masked_image = st.session_state["bg_image"].copy()
            masked_image.paste(overlay_masks, (0, 0), overlay_masks)

            # json data
            objects = pd.json_normalize(
                canvas_result.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")

            # make masks
            color2boxes = {}
            for _, row in objects.iterrows():
                color = row["stroke"]
                left, top, width, height = (
                    row["left"],
                    row["top"],
                    row["width"],
                    row["height"],
                )
                right, bottom = left + width, top + height
                if not inv_colors_map[color] in color2boxes:
                    color2boxes[inv_colors_map[color]] = []
                color2boxes[inv_colors_map[color]].append((left, top, right, bottom))
            color2mask = {}
            for color, boxes in color2boxes.items():
                mask = Image.new("L", (512, 512), 0)
                for box in boxes:
                    mask.paste(255, box)
                color2mask[color] = pil_to_tensor(mask)

            # generate prompts
            st.session_state["masks"] = list(color2mask.values())
            st.session_state["color_names"] = list(color2mask.keys())
            st.session_state["prompts"] = generate_prompts(
                masked_image, st.session_state["color_names"]
            )
            for k, v in st.session_state["prompts"].items():
                st.session_state["prompts"][k] = tr_enit_pipe(v)[0]["translation_text"]
            st.session_state["disable_inpaint"] = False
        else:
            st.warning("Please draw on the canvas.")

if "prompts" in st.session_state:
    cols = st.columns(len(st.session_state["color_names"]))
    for i, color in enumerate(st.session_state["color_names"]):
        with cols[i]:
            st.session_state["prompts"][color] = st.text_area(
                f"Prompt for {colors_str_map[color]}",
                st.session_state["prompts"][color],
            )

with r2_col2:
    if st.button(
        "Inpaint!", type="primary", disabled=st.session_state["disable_inpaint"]
    ):
        # Stack masks along a new dimension (dim=0), resulting in a tensor of shape (N, 1, 512, 512)
        stacked_masks = torch.stack(st.session_state["masks"], dim=0)
        # Take the element-wise max across the first dimension (N), resulting in shape (1, 512, 512)
        final_mask, _ = torch.max(stacked_masks, dim=0)
        freestyle_layout_mask = get_freestyle_layout_mask(
            st.session_state["masks"], max_concepts=5
        )
        eng_prompts = [tr_iten_pipe(prompt)[0]["translation_text"] for prompt in st.session_state["prompts"].values()]
        input_ids, freestyle_attention_mask = get_freestyle_text_inputs(
            eng_prompts, mm_inpainting_pipe.tokenizer
        )
        inpainted = mm_inpainting_pipe(
            input_ids=input_ids.unsqueeze(0).to(device),
            freestyle_attention_mask=freestyle_attention_mask.unsqueeze(0).to(device),
            freestyle_layout_mask=freestyle_layout_mask.unsqueeze(0).to(device),
            image=st.session_state["bg_image"],
            mask_image=final_mask,
        )
        with r1_col2:
            st.image(inpainted.images[0])
