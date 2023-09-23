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

from utils import create_hed

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

controlnet = ControlNetModel.from_pretrained(args.controlnet_path)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    args.bas_sd_model,
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
    
    random_seed = seed

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(init_img, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_img = Image.fromarray(image)


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
    
    random_seed = seed


    for i, file in enumerate(file_urls):

        init_img = cv2.imread(file)
         
        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(init_img, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_img = Image.fromarray(image)

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

    if(file_index < 0):
        file_index = 0
    else:
        file_index+=1

    img_a = cv2.imread(os.path.join(args.output_dir,f'train_A/{file_index:04d}.jpg'))
    img_b = cv2.imread(os.path.join(args.output_dir,f'train_B/{file_index:04d}.jpg'))

    image_pair = np.hstack([img_a,img_b])

    image_pair = cv2.cvtColor(np.uint8(image_pair),cv2.COLOR_BGR2RGB)
     
    return image_pair

def load_previous_img(directory):

    global file_index

    directory_path = directory

    file_urls = sorted(
        [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))],
        key=str.casefold,  # This will sort the URLs in a case-insensitive manner
    )


    if(file_index == len(file_urls)):
        file_index = len(file_urls)
    else:
        file_index-=1

    img_a = cv2.imread(os.path.join(args.output_dir,f'train_A/{file_index:04d}.jpg'))
    img_b = cv2.imread(os.path.join(args.output_dir,f'train_B/{file_index:04d}.jpg'))

    image_pair = np.hstack([img_a,img_b])

    image_pair = cv2.cvtColor(np.uint8(image_pair),cv2.COLOR_BGR2RGB)
     
    return image_pair

def regenerate_pair(review_data_dir,prompt,negative_prompt, controlnet_str,img2img_str,steps,cfg,seed,index):

    # global file_index
    file_index = index
        
    random_seed = seed

    prompt_embeds = compel_proc(prompt)
    negative_prompt_embeds = compel_proc(negative_prompt)

    if(file_index>0):
        img_0 = os.path.join(review_data_dir,f'train_A/{file_index-1:04d}.jpg')
        img_0_b = os.path.join(review_data_dir,f'train_B/{file_index-1:04d}.jpg')
        img_0 = cv2.imread(img_0)
        img_0_b = cv2.imread(img_0_b)
   
    img_1 = os.path.join(review_data_dir,f'train_A/{file_index+1:04d}.jpg')
    img_1_b = os.path.join(review_data_dir,f'train_B/{file_index+1:04d}.jpg')

    img_a = os.path.join(review_data_dir,f'train_A/{file_index:04d}.jpg')
    init_img = cv2.imread(img_a)

  
    img_1 = cv2.imread(img_1)
    img_1_b = cv2.imread(img_1_b)
        
    low_threshold = 100
    high_threshold = 200


    image = cv2.Canny(init_img, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_img = Image.fromarray(image)

    out_img = pipe(prompt_embeds=prompt_embeds,
                  negative_prompt_embeds = negative_prompt_embeds,
                  controlnet_conditioning_image = canny_img,
                  controlnet_conditioning_scale = controlnet_str,
                  image= init_img,
                  strength = img2img_str,
                  num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                  guidance_scale = cfg).images[0]
    

    

    controlnet_img = np.uint8(init_img)
    controlnet_img_pair = cv2.cvtColor(np.uint8(controlnet_img),cv2.COLOR_BGR2RGB)
    out_img = cv2.cvtColor(np.uint8(out_img),cv2.COLOR_BGR2RGB)

    
    image_pair = np.hstack([controlnet_img_pair,out_img])
    next_pair = np.hstack([img_1,img_1_b])

    if(file_index>0):
        prev_pair = np.hstack([img_0,img_0_b])
        output = np.vstack([prev_pair,image_pair,next_pair])
    else:
        output = np.vstack([image_pair,next_pair])

    

    cv2.imwrite(os.path.join(review_data_dir,f'train_A/{file_index:04d}.jpg'),controlnet_img)
    cv2.imwrite(os.path.join(review_data_dir,f'train_B/{file_index:04d}.jpg'),out_img)
    
    return output

# def save_config():
#     #saves the config

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

                    # controlnet_output = gr.Gallery(
                    #     columns = [1],
                    #     object_fit='fill',
                    #     show_label=True
                    # )
                    controlnet_output = gr.Image()

        with gr.Row():

                controlnet_submit = gr.Button("Test")
                batch_submit = gr.Button("Batch")

    with gr.Tab("review pairs"):
         
        with gr.Row():

            with gr.Column():

                review_data_dir = gr.Textbox(label="data dir")

                review_prompt_input = gr.Textbox(label="prompt")
                review_negative_prompt_input = gr.Textbox(label="negative prompt")
                review_conditioning_scale_input = gr.Slider(0, 1, 
                    value=0.8, label="controlnet_conditioning_scale")
                review_steps_input = gr.Slider(0, 150, value=20,
                    label="number of diffusion steps")
                review_cfg_input = gr.Slider(0,30,value=3.5,label="cfg scale")
                index = gr.Slider(0, 1000, value=00,
                    label="index")

                
                review_inputs = [
                    review_data_dir,
                    review_prompt_input,
                    review_negative_prompt_input,
                    review_conditioning_scale_input,
                    review_steps_input,
                    review_cfg_input,
                    index
                ]


            with gr.Column():

                review_output = gr.Image()

        with gr.Row():

            load_next = gr.Button("load next")
            review_submit = gr.Button("regenerate")
            load_previous = gr.Button("load previous")



    controlnet_submit.click(generate_image_pair,inputs=controlnet_inputs,outputs=controlnet_output)
    batch_submit.click(generate_batch,inputs=batch_inputs,outputs=controlnet_output)

    review_submit.click(regenerate_pair,inputs=review_inputs,outputs=review_output)
    load_next.click(load_next_img,inputs=review_data_dir,outputs=review_output)
    load_previous.click(load_previous_img,inputs=review_data_dir,outputs=review_output)

if __name__ == "__main__":
    demo.launch(share=True,debug=True)