import functools

from flask import Flask, request, redirect, url_for
from flask import Blueprint, request, render_template, url_for, current_app, session, g, flash, jsonify
from werkzeug.security import generate_password_hash
from werkzeug.utils import redirect

from main import args
from models import update_crop_image, get_images, get_detail, get_summary, get_graph
from form import SupervisingForm

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

@bp.route('/detail/', methods=('GET', 'POST'))
def detail():
    form = SupervisingForm()
    if request.method == 'POST' and form.validate_on_submit():
        filename = form.filename.data
        order = form.order.data
        target = form.target.data
        request_data = {'path_folder':args.result_folder + filename.split('.')[0], 'order':int(order), 'target':target}
        update_crop_image(request_data)

    page = int(request.args.get('page', 1))
    filename = request.args.get('filename', None)
    paging, crop_imgs = get_detail(page=page, filename=filename)
    return render_template('train/detail.html', **locals())

@bp.route('/summary/')
def summary():
    page = int(request.args.get('page', 1))
    # filename = request.args.get('filename', None)
    paging, img_list, total = get_summary(page=page)
    return render_template('train/summary.html', **locals())

@bp.route('/graph/')
def graph():
    xy_list = get_graph()
    return render_template('train/graph.html', **locals())






