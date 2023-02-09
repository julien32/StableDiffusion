from flask import Blueprint, render_template, request, flash, redirect, url_for
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import monkeypatch_lora, tune_lora_scale, patch_pipe
from flask_login import login_required, current_user
import numpy as np
from PIL import Image
import glob
import os
from lora_diffusion.cli_lora_pti import train as train_lora_source

train_lora = Blueprint('train_lora', __name__)
root_dir = "/home/lamparter/stableDiffusion"

def run_train_lora(imagepath, token, template):
    train_lora_source(
            instance_data_dir = imagepath,
            pretrained_model_name_or_path = '{0}/models/stable-diffusion-v1-5'.format(root_dir),
            output_dir = '{0}/StableDiffusionFlask/static/trained_models'.format(root_dir),
            train_text_encoder = True,
            resolution = 256,
            train_batch_size = 1,
            gradient_accumulation_steps = 4,
            scale_lr = True,
            learning_rate_unet = 1e-4,
            learning_rate_text = 1e-5,
            learning_rate_ti = 5e-4,
            color_jitter = True,
            lr_scheduler = "linear",
            lr_warmup_steps = 0,
            placeholder_tokens = token,
            use_template = template,
            save_steps = 100,
            max_train_steps_ti = 1000,
            max_train_steps_tuning = 1000,
            perform_inversion = True,
            clip_ti_decay = True,
            weight_decay_ti = 0.000,
            weight_decay_lora = 0.001,
            continue_inversion = True,
            continue_inversion_lr = 1e-4,
            device = "cuda:0",
            lora_rank = 1
    )


@train_lora.route("/train_lora", methods=['GET', 'POST'])
@login_required
def train():
    user_id = current_user.email
    if request.method == 'POST':
        save_path = '/home/lamparter/stableDiffusion/StableDiffusionFlask/static/upload_folder'
        files=request.files.getlist("files[]")
        for file in files:
            file.save(os.path.join(save_path, file.filename))
        run_train_lora(save_path, request.form['token'], request.form['type'])
        return "Hallo"
    return render_template("train_lora.html", user=current_user)

