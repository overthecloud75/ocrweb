import os
from logging.config import dictConfig
from flask import Flask
import argparse

# CRAFT
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--static_folder', default='static/', type=str, help='folder path to web images')
parser.add_argument('--result_folder', default='results/', type=str, help='folder path to input images')
parser.add_argument('--upload_folder', default='upload/', type=str, help='folder path to upload images')
parser.add_argument('--result_width', default=500, type=int, help='image size of result image')
parser.add_argument('--crop_height', default=20, type=int, help='web image size of crop image')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

if not os.path.isdir(args.static_folder):
    os.makedirs(args.static_folder)
if not os.path.isdir(args.upload_folder):
    os.makedirs(args.upload_folder)

def create_app():
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        }
    })

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(32)

    from views import main_views, upload_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(upload_views.bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', debug=True)
