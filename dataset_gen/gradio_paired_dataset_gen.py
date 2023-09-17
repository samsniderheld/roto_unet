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

controlnet = ControlNetModel.from_pretrained(args.controlnet_path)
control_net_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    args.base_sd_model,
    controlnet=controlnet,
    safety_checker=None,
).to('cuda')
control_net_pipe.scheduler = UniPCMultistepScheduler.from_config(control_net_pipe.scheduler.config)

file_index = 0

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

def generate_batch(control_net_prompt,controlnet_conditioning_scale,steps,cfg,batch_dir):
    
    directory_path = batch_dir

    file_urls = sorted(
        [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))],
        key=str.casefold,  # This will sort the URLs in a case-insensitive manner
    )

    negative_prompt = f'poor quality, low quality'
    
    random_seed = random.randrange(0,100000)


    for i, file in enumerate(file_urls):

        controlnet_img = cv2.imread(file)
         
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

        controlnet_img = np.uint8(controlnet_img)
        out_img = cv2.cvtColor(np.uint8(out_img),cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(args.output_dir,f'train_A/{i:04d}.jpg'),controlnet_img)
        cv2.imwrite(os.path.join(args.output_dir,f'train_B/{i:04d}.jpg'),out_img)
    
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

def regenerate_pair(review_data_dir,review_prompt_input, negative_prompt_input, review_conditioning_scale_input,review_steps_input,review_cfg_input):

    global file_index
        
    random_seed = random.randrange(0,100000)

    img_a = os.path.join(review_data_dir,f'train_A/{file_index:04d}.jpg')

    controlnet_img = cv2.imread(img_a)
        
    low_threshold = 100
    high_threshold = 200

    controlnet_img = cv2.resize(controlnet_img,(512,512))

    image = cv2.Canny(controlnet_img, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    out_img = control_net_pipe(prompt=review_prompt_input,
                    negative_prompt = negative_prompt_input,
                    image= canny_image,
                    controlnet_conditioning_scale=review_conditioning_scale_input,
                    height=512,
                    width=512,
                    num_inference_steps=review_steps_input, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = review_cfg_input).images[0]
    

    

    controlnet_img = np.uint8(controlnet_img)
    controlnet_img_pair = cv2.cvtColor(np.uint8(controlnet_img),cv2.COLOR_BGR2RGB)
    out_img = cv2.cvtColor(np.uint8(out_img),cv2.COLOR_BGR2RGB)


    image_pair = np.hstack([controlnet_img_pair,out_img])

    cv2.imwrite(os.path.join(review_data_dir,f'train_A/{file_index:04d}.jpg'),controlnet_img)
    cv2.imwrite(os.path.join(review_data_dir,f'train_B/{file_index:04d}.jpg'),out_img)
    
    return image_pair

# def save_config():
#     #saves the config

with gr.Blocks() as demo:

    with gr.Tab("generate pairs"):

        with gr.Row():

                with gr.Column():
                    controlnet_prompt_input = gr.Textbox(label="prompt")
                    controlnet_input_img = gr.Image(label="input img")
                    controlnet_conditioning_scale_input = gr.Slider(0, 1, 
                        value=0.8, label="controlnet_conditioning_scale")
                    controlnet_steps_input = gr.Slider(0, 150, value=20,
                        label="number of diffusion steps")
                    controlnet_cfg_input = gr.Slider(0,30,value=3.5,label="cfg scale")

                    batch_input = gr.Textbox(label="batch_directory")

                    controlnet_inputs = [
                        controlnet_prompt_input,
                        controlnet_input_img,
                        controlnet_conditioning_scale_input,
                        controlnet_steps_input,
                        controlnet_cfg_input,
                    ]

                    batch_inputs = [
                        controlnet_prompt_input,
                        controlnet_conditioning_scale_input,
                        controlnet_steps_input,
                        controlnet_cfg_input,
                        batch_input
                    ]

                with gr.Column():

                    controlnet_output = gr.Gallery(
                        columns = [1],
                        object_fit='fill',
                        show_label=True
                    )

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

                review_inputs = [
                    review_data_dir,
                    review_prompt_input,
                    review_negative_prompt_input,
                    review_conditioning_scale_input,
                    review_steps_input,
                    review_cfg_input,
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