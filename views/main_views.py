import functools

from flask import Flask, request, redirect, url_for
from flask import Blueprint, request, render_template, url_for, current_app, session, g, flash, jsonify
from werkzeug.security import generate_password_hash
from werkzeug.utils import redirect

from main import args
from models import update_crop_image, get_images, get_detail, get_summary, get_statistics, get_loss
from form import SupervisingForm

# blueprint
bp = Blueprint('main', __name__, url_prefix='/')

def confidence():
    pass

@bp.route('/')
def index():
    return render_template('base.html')

@bp.route('/summary/')
def summary():
    page = int(request.args.get('page', 1))
    # filename = request.args.get('filename', None)
    paging, img_list, total = get_summary(page=page)
    return render_template('train/summary.html', **locals())

@bp.route('/prediction/')
def prediction():
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

@bp.route('/statistics/')
def statistics():
    xy_list, path_list, confidence_list = get_statistics()
    return render_template('train/statistics.html', **locals())

@bp.route('/loss/')
def loss():
    loss_data = get_loss()
    model_list = []
    epoch_list = []
    loss_list = {}
    accuracy_list = {}
    colors = ['#ff0000', '#0000ff', '#f56d798', '#ff8397', '#6970d5']
    for model in loss_data:
        epochs = []
        model_list.append(model)
        loss_list[model] = []
        accuracy_list[model] = []
        for data in loss_data[model]:
            epochs.append(data['epoch'])
            loss_list[model].append(data['loss'])
            accuracy_list[model].append(data['accuracy'])
        if epoch_list:
            if len(epoch_list) < len(epochs):
                epoch_list = epochs
        else:
            epoch_list = epochs
    return render_template('train/loss.html', **locals())






