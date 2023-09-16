import argparse
import os
import random
import time

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageFilter
import torch

from diffusers.utils import load_image

from diffusers import (ControlNetModel,StableDiffusionControlNetPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
)
from diffusers import UniPCMultistepScheduler


def parse_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A Hugging Face Diffusers Pipeline for generating a rotoscope paired dataset"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text

    parser.add_argument(
        '--input_dir', type=str, default='the base dataset dir', 
        help='The directory for all the output results.')

    parser.add_argument(
        '--output_dir', type=str, default='paired data outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--base_sd_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using')
    parser.add_argument(
        '--controlnet_path', type=str, default='lllyasviel/sd-controlnet-canny', 
        help='The controlnet model we are using.')

    return parser.parse_args()

args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir,'train_A'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir,'train_B'), exist_ok=True)

controlnet = ControlNetModel.from_pretrained(args.controlnet_path)
control_net_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    args.base_sd_model,
    controlnet=controlnet,
    safety_checker=None,
).to('cuda')
control_net_pipe.scheduler = UniPCMultistepScheduler.from_config(control_net_pipe.scheduler.config)

def generate_image_pair(control_net_prompt, controlnet_img,controlnet_conditioning_scale,steps,cfg):
    #generates pair


    negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'
    
    random_seed = random.randrange(0,100000)

    # controlnet_img = Image.fromarray(controlnet_img)

    low_threshold = 100
    high_threshold = 200

    controlnet_img = cv2.resize(controlnet_img,(512,512))

    image = cv2.Canny(controlnet_img, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    out_img = control_net_pipe(prompt=control_net_prompt,
                    negative_prompt = negative_prompt,
                    image= canny_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    height=512,
                    width=512,
                    num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]
    

    image_pair = np.hstack([controlnet_img,out_img])
    
    return image_pair

    #generates the burger

# def save_config():
#     #saves the config

with gr.Blocks() as demo:

    #texture gen row
    with gr.Row():

            with gr.Column():
                controlnet_prompt_input = gr.Textbox(label="prompt")
                controlnet_input_img = gr.Image(label="input img")
                controlnet_conditioning_scale_input = gr.Slider(0, 1, 
                    value=0.8, label="controlnet_conditioning_scale")
                controlnet_steps_input = gr.Slider(0, 150, value=20,
                    label="number of diffusion steps")
                controlnet_cfg_input = gr.Slider(0,30,value=3.5,label="cfg scale")

                controlnet_inputs = [
                    controlnet_prompt_input,
                    controlnet_input_img,
                    controlnet_conditioning_scale_input,
                    controlnet_steps_input,
                    controlnet_cfg_input,
                ]

            with gr.Column():

                controlnet_output = gr.Image()

    with gr.Row():

            controlnet_submit = gr.Button("Submit")



    controlnet_submit.click(generate_image_pair,inputs=controlnet_inputs,outputs=controlnet_output)


if __name__ == "__main__":
    demo.launch(share=True,debug=True)