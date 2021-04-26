import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

from flask import Blueprint, request, render_template, url_for, current_app, session, g, flash, jsonify
from main import args
from models import get_images
from werkzeug.security import generate_password_hash
from werkzeug.utils import redirect
import functools
from PIL import Image

# blueprint
bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('base.html')

@bp.route('/train/')
def train():
    page = int(request.args.get('page', 1))
    paging, img_list, crop_imgs = get_images(page=page)
    return render_template('train/train.html', **locals())



