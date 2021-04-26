import os
from flask import Flask, request, redirect, url_for
from flask import Blueprint, request, render_template, url_for, current_app, session, g, flash, jsonify
from main import args
from werkzeug.utils import redirect
import datetime

import torch
import torch.backends.cudnn as cudnn
import cv2
from craft import CRAFT
import pipeline, utils, imgproc
from models import update_image


# blueprint
bp = Blueprint('upload', __name__, url_prefix='/upload')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'jfif', 'xlsx'])

def execute_net():
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(pipeline.copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(pipeline.copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet

        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(pipeline.copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(pipeline.copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
        refine_net.eval()
        args.poly = True
    return net, refine_net

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def execute_ocr(filename, file_path):
    image = imgproc.loadImage(file_path)
    bboxes, polys, det_scores = \
        pipeline.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly,
                          args, refine_net)
    bbox_scores = []
    for box_num in range(len(bboxes)):
        bbox_scores.append((str(det_scores[box_num]), bboxes[box_num]))
    box_filename, height, width = utils.saveResult(file_path, image[:, :, ::-1], polys, adjust_width=args.result_width, base_dir=args.static_folder, dirname=args.result_folder)
    update_image({'path':args.result_folder + box_filename, 'height':height, 'width':width})
    image = cv2.imread(file_path)
    pipeline.generate_words(filename, bbox_scores, image, base_dir=args.static_folder, dirname=args.result_folder)

@bp.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            ext = file.filename.split('.')[-1]
            filename = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + '.' + ext
            file_path = os.path.join(args.upload_folder, filename)
            file.save(file_path)
            execute_ocr(filename, file_path)
            return redirect(url_for('main.train', filename=filename))
    return render_template('upload.html')

net, refine_net = execute_net()