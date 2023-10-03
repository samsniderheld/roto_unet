import sys
sys.path.append('../')
import argparse
import os
import random
import time

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageFilter
import torch

from compel import Compel

from diffusers import (ControlNetModel,StableDiffusionImg2ImgPipeline)

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
        '--output_dir', type=str, default='paired_data_outputs', 
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
os.makedirs(os.path.join(args.output_dir,'paired'), exist_ok=True)

controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    args.base_sd_model,
    custom_pipeline="stable_diffusion_controlnet_img2img",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
).to('cuda')

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

file_index = 0

def generate_image_pair(prompt, negative_prompt, init_img, controlnet_str, img2img_str,steps,cfg,seed):
    #generates pair

    prompt_embeds = compel_proc(prompt)
    negative_prompt_embeds = compel_proc(negative_prompt)
    
    random_seed = int(seed)

    low_threshold = 100
    high_threshold = 200

    init_img = cv2.resize(init_img,(512,512))

    image = cv2.Canny(init_img, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    canny_img = Image.fromarray(image)

    init_img = Image.fromarray(init_img)

    out_img = pipe(prompt_embeds=prompt_embeds,
                  negative_prompt_embeds = negative_prompt_embeds,
                  controlnet_conditioning_image = canny_img,
                  controlnet_conditioning_scale = controlnet_str,
                  image= init_img,
                  strength = img2img_str,
                  num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                  guidance_scale = cfg).images[0]


    image_pair = np.hstack([init_img,canny_img,out_img])
    
    return image_pair

def generate_batch(prompt,negative_prompt,controlnet_str,img2img_str,steps,cfg,seed, batch_dir):
    
    directory_path = batch_dir

    file_urls = sorted(
        [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))],
        key=str.casefold,  # This will sort the URLs in a case-insensitive manner
    )

    prompt_embeds = compel_proc(prompt)
    negative_prompt_embeds = compel_proc(negative_prompt)
    
    random_seed = int(seed)

    for i, file in enumerate(file_urls):

        init_img = cv2.imread(file)
        init_img = cv2.resize(init_img,(512,512))
         
        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(init_img, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_img = Image.fromarray(image)

        init_img = Image.fromarray(init_img)

        out_img = pipe(prompt_embeds=prompt_embeds,
                  negative_prompt_embeds = negative_prompt_embeds,
                  controlnet_conditioning_image = canny_img,
                  controlnet_conditioning_scale = controlnet_str,
                  image= init_img,
                  strength = img2img_str,
                  num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                  guidance_scale = cfg).images[0]
        
        image_pair = np.hstack([init_img,canny_img,out_img])

        init_img = np.uint8(init_img)
        out_img = cv2.cvtColor(np.uint8(out_img),cv2.COLOR_BGR2RGB)
        out_pair = np.uint8(image_pair)
        
        cv2.imwrite(os.path.join(args.output_dir,f'train_A/{i:04d}.jpg'),init_img)
        cv2.imwrite(os.path.join(args.output_dir,f'train_B/{i:04d}.jpg'),out_img)
        cv2.imwrite(os.path.join(args.output_dir,f'paired/{i:04d}.jpg'),out_pair)
    
    return image_pair

def load_next_img(directory):

    global file_index

    directory_path = os.path.join(directory,"paired")

    file_urls = sorted(
        [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))],
        key=str.casefold,  # This will sort the URLs in a case-insensitive manner
    )

    if(file_index == len(file_urls)):
        file_index = len(file_urls)
    else:
        file_index+=1

    path = file_urls[file_index]
    print(path)
    pair = cv2.imread(path)

    image_pair = cv2.cvtColor(np.uint8(pair),cv2.COLOR_BGR2RGB)
     
    return image_pair

def load_previous_img(directory):

    global file_index

    directory_path = os.path.join(directory,"paired")

    file_urls = sorted(
        [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))],
        key=str.casefold,  # This will sort the URLs in a case-insensitive manner
    )

    if(file_index < 0):
        file_index = 0
    else:
        file_index-=1    

    path = file_urls[file_index]
    print(path)
    pair = cv2.imread(path)
    image_pair = cv2.cvtColor(np.uint8(pair),cv2.COLOR_BGR2RGB)
     
    return image_pair

def delete_pair(review_data_dir):

    global file_index
    
    paired_path = os.path.join(review_data_dir,"paired")
    train_A_path = os.path.join(review_data_dir,"train_A")
    train_B_path = os.path.join(review_data_dir,"train_B")

    file_urls_paired = sorted(
        [os.path.join(paired_path, f) for f in os.listdir(paired_path) if os.path.isfile(os.path.join(paired_path, f))],
        key=str.casefold,  # This will sort the URLs in a case-insensitive manner
    )

    file_urls_train_A = sorted(
        [os.path.join(train_A_path, f) for f in os.listdir(train_A_path) if os.path.isfile(os.path.join(train_A_path, f))],
        key=str.casefold,  # This will sort the URLs in a case-insensitive manner
    )

    file_urls_train_B = sorted(
        [os.path.join(train_B_path, f) for f in os.listdir(train_B_path) if os.path.isfile(os.path.join(train_B_path, f))],
        key=str.casefold,  # This will sort the URLs in a case-insensitive manner
    )
    os.remove(file_urls_paired[file_index])
    os.remove(file_urls_train_A[file_index])
    os.remove(file_urls_train_B[file_index])

    return None

with gr.Blocks() as demo:

    with gr.Tab("generate pairs"):

        with gr.Row():

                with gr.Column():
                    controlnet_prompt_input = gr.Textbox(label="prompt")
                    negative_control_net_prompt = gr.Textbox(label="negative prompt")
                    controlnet_input_img = gr.Image(label="input img")
                    controlnet_conditioning_scale_input = gr.Slider(0, 1, 
                        value=0.8, label="controlnet_conditioning_scale")
                    img2img_str = gr.Slider(0, 1, 
                        value=0.8, label="img2img strength")
                    controlnet_steps_input = gr.Slider(0, 150, value=20,
                        label="number of diffusion steps")
                    controlnet_cfg_input = gr.Slider(0,30,value=3.5,label="cfg scale")
                    seed = gr.Number()

                    batch_input = gr.Textbox(label="batch_directory")

                    controlnet_inputs = [
                        controlnet_prompt_input,
                        negative_control_net_prompt,
                        controlnet_input_img,
                        controlnet_conditioning_scale_input,
                        img2img_str,
                        controlnet_steps_input,
                        controlnet_cfg_input,
                        seed
                    ]

                    batch_inputs = [
                        controlnet_prompt_input,
                        negative_control_net_prompt,
                        controlnet_conditioning_scale_input,
                        img2img_str,
                        controlnet_steps_input,
                        controlnet_cfg_input,
                        seed,
                        batch_input
                    ]

                with gr.Column():

                    controlnet_output = gr.Image()

        with gr.Row():

                controlnet_submit = gr.Button("Test")
                batch_submit = gr.Button("Batch")

    with gr.Tab("review pairs"):
         
        with gr.Row():

            with gr.Column():

                review_data_dir = gr.Textbox(label="data dir")


            with gr.Column():

                review_output = gr.Image()

        with gr.Row():

            load_next = gr.Button("load next")
            delete_button = gr.Button("delete")
            load_previous = gr.Button("load previous")



    controlnet_submit.click(generate_image_pair,inputs=controlnet_inputs,outputs=controlnet_output)
    batch_submit.click(generate_batch,inputs=batch_inputs,outputs=controlnet_output)
    delete_button.click(delete_pair,inputs=review_data_dir,outputs=review_output)
    load_next.click(load_next_img,inputs=review_data_dir,outputs=review_output)
    load_previous.click(load_previous_img,inputs=review_data_dir,outputs=review_output)

if __name__ == "__main__":
    demo.launch(share=True,debug=True)