import os
import datetime

from flask import Blueprint, request, render_template, url_for, jsonify 
from werkzeug.utils import redirect

import torch
import torch.backends.cudnn as cudnn

from main import args
from recognition import pipeline
from recognition.craft import CRAFT
from recognition.model import Model
from recognition.utils import CTCLabelConverter, AttnLabelConverter

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# blueprint
bp = Blueprint('upload', __name__, url_prefix='/upload')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'jfif', 'xlsx'])

def execute_detection_model():
    # load net
    net = CRAFT()

    print('Loading detection model from %s' %args.detection_model)
    if args.cuda:
        net.load_state_dict(pipeline.copyStateDict(torch.load(args.detection_model)))
    else:
        net.load_state_dict(pipeline.copyStateDict(torch.load(args.detection_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from recognition.refinenet import RefineNet

        refine_net = RefineNet()
        print('Loading refiner model from %s' %args.refiner_model)
        if args.cuda:
            refine_net.load_state_dict(pipeline.copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(pipeline.copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
        refine_net.eval()
        args.poly = True
    return net, refine_net

def excute_recognition_model():
    """Open csv file wherein you are going to write the Predicted Words"""
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    model = Model(args)
    model.to(device)
    #model = torch.nn.DataParallel(model).to(device)

    # load model
    print('Loading recognition model from %s' %args.recognition_model)
    model.load_state_dict(torch.load(args.recognition_model, map_location=device))

    return model, converter

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ocr(file):
    prediction = []
    if file and allowed_file(file.filename):
        ext = file.filename.split('.')[-1]
        filename = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + '.' + ext
        file_path = os.path.join(args.upload_folder, filename)
        file.save(file_path)
        prediction = pipeline.execute_ocr(filename, file_path, net=net, refine_net=refine_net, model=model, converter=converter)
    return prediction

@bp.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        ocr(file)
        return redirect(url_for('main.train'))
    return render_template('upload.html')

@bp.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    prediction = ocr(file)
    return jsonify(prediction)

net, refine_net = execute_detection_model()
model, converter = excute_recognition_model()
