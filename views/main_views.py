import functools

from flask import Flask, request, redirect, url_for
from flask import Blueprint, request, render_template, url_for, current_app, session, g, flash, jsonify
from werkzeug.security import generate_password_hash
from werkzeug.utils import redirect

from models import get_images, get_detail

# blueprint
bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('base.html')

@bp.route('/train/')
def train():
    page = int(request.args.get('page', 1))
    # filename = request.args.get('filename', None)
    paging, img_list, crop_imgs = get_images(page=page)
    return render_template('train/train.html', **locals())

@bp.route('/detail/')
def detail():
    filename = request.args.get('filename', None)
    crop_imgs = get_detail(filename=filename)
    return render_template('train/detail.html', **locals())




