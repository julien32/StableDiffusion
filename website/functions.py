import subprocess
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user

functions = Blueprint('functions', __name__)

@functions.route('/run_text2Img/<string:prompt>')
@login_required
def run_text2Img(prompt):
    command = prompt
    output = subprocess.run(command, shell=True, capture_output=True)
    print(output)

    return output.stdout