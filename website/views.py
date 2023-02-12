from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note, User
from .functions import get_random_images, get_users_trained_models
from . import db
import json
import os

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    user_trained_models = get_users_trained_models()
    random_images = get_random_images()
    
    return render_template("home.html", user=current_user, random_images=random_images, user_trained_models=user_trained_models)

@views.route('/generated_images', methods=['GET', 'POST'])
@login_required
def generated_images():

    directory = '/home/lamparter/stableDiffusion/StableDiffusionFlask/static/generated_images/'

    files = [os.path.basename(file) for file in os.listdir(directory) if file.endswith('.png')]
    print(files)
    
    return render_template("generated_images.html", user=current_user, gif_images = files)

@views.route('/compare', methods=['GET', 'POST'])
@login_required
def generated_images_compare():
    
    return render_template("generated_images_compare.html", user=current_user)

@views.route('/prompt_generator', methods=['GET', 'POST'])
@login_required
def prompt_generator():
    
    return render_template("prompt_generator.html", user=current_user)