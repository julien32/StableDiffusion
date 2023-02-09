import subprocess
import torch
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from lora_diffusion import monkeypatch_lora, tune_lora_scale, patch_pipe
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import numpy as np
from PIL import Image
import glob
import os
from .models import Img, User, Model
from . import db
import json
import random

functions = Blueprint('functions', __name__)

root_dir = "/home/lamparter/stableDiffusion"

@functions.route('/run_text2Img/<string:prompt>')
@login_required
def run_text2Img(prompt):
    command = prompt
    output = subprocess.run(command, shell=True, capture_output=True)
    print(output)

    return "hello"

@functions.route('/run_lora/<string:prompt>/<string:model_name>')
@login_required
def run_lora(prompt, model_name):
    user = current_user
    models = ['stable-diffusion-v1-5'.format(root_dir)]
    models += get_models_of_user(user)
    print(request.data)
    # model_id = request.form["model_id"]
    model_id = "stable-diffusion-v1-5"
    create_image(prompt, model_name, user)
    
    
    flash('Image created!', category='success')


    return "finished"


def get_models_of_user(user):
    models = glob.glob('{0}/models/{1}/*'.format(root_dir, user))
    models = [model.split('/')[-1] for model in models]
    return models

def get_random_images():
    images = glob.glob('{0}/StableDiffusionFlask/static/generated_images/*'.format(root_dir))
    images.sort(key=os.path.getmtime)
    images.reverse()
    paths = ['/static/generated_images/{0}'.format(path.split("/")[-1]) for path in images]
    random.shuffle(paths)
    
    object_images = []
    for path in paths:
        prompt =  path.split("_")[1].split("/")[1]
        image = Img(image_path=path, prompt=prompt, user_id=current_user.id)
        object_images.append(image)
    
    return object_images

def create_image(prompt, model_name, user):
    pipe = StableDiffusionPipeline.from_pretrained('{0}/models/stable-diffusion-v1-5'.format(root_dir), torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if model_name != 'stable-diffusion-v1-5':
        patch_pipe(pipe, '{0}/StableDiffusionFlask/static/trained_models/{1}'.format(root_dir, model_name), patch_text=True, patch_ti=True, patch_unet=True)
        tune_lora_scale(pipe.unet, 1.00)
        
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
    imagename = '{0}_{1}_{2}.png'.format(prompt, user.id, random.randint(1, 10000000000))
    image_save_path = "static/generated_images/" + imagename
    
    add_note_to_db(imagename, prompt, model_name)
    image.save(image_save_path)
    


def add_note_to_db(path, prompt, model_name):
    new_image = Img(image_path=path, prompt=prompt, model=model_name, user_id=current_user.id)
    db.session.add(new_image)
    db.session.commit()
    
def add_model_to_db(name):
    new_model = Model(name=name)
    db.session.add(new_model)
    db.session.commit()
    