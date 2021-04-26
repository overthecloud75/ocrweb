import os
from werkzeug.security import check_password_hash

import datetime

from utils import paginate
from main import args
from views.config import page_default

from pymongo import MongoClient
mongoClient = MongoClient('mongodb://localhost:27017/')
db = mongoClient['OcrWeb']

# users
def post_signUp(request_data):
    collection = db['users']
    user_data = collection.find_one(filter={'email': request_data['email']})
    error = None
    if user_data:
        error = '이미 존재하는 사용자입니다.'
    else:
        user_data = collection.find_one(sort=[('create_time', -1)])
        if user_data:
            user_id = user_data['user_id'] + 1
        else:
            user_id = 1
        request_data['create_time'] = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        request_data['user_id'] = user_id
        collection.insert(request_data)
    return error

def post_login(request_data):
    collection = db['users']
    error = None
    user_data = collection.find_one(filter={'email': request_data['email']})
    if not user_data:
        error = "존재하지 않는 사용자입니다."
    elif not check_password_hash(user_data['password'], request_data['password']):
        error = "비밀번호가 올바르지 않습니다."
    return error, user_data

# images
def update_image(request_data):
    collection = db['images']
    collection.insert_one(request_data)

def get_images(page=1):
    collection = db['images']

    per_page = page_default['per_page']
    offset = (page - 1) * per_page

    data_list = collection.find(sort=[('path', -1)])
    count = data_list.count()
    data_list = data_list.limit(per_page).skip(offset)
    paging = paginate(page, per_page, count)
    collection = db['crop_images']
    img_list = []
    crop_imgs = {}
    for data in data_list:
        path = data['path']                    # results/20210425214433.jpg
        img_list.append({'path':path, 'name':path.split('/')[1]})
        path_folder = path.split('.')[0]       # results/20210425214433
        static_path = args.static_folder + path_folder  # static/results/20210425214433
        if os.path.isdir(static_path):
            crop_imgs[path] = []
            crop_list = collection.find({'path_folder':path_folder}, sort=[('order', 1)])
            for crop in crop_list:
                width = int(crop['width'] / crop['height'] * args.crop_height)
                if width > args.crop_width:
                    width = args.crop_width
                crop_imgs[path].append({'path':path_folder + '/' + crop['name'], 'height':args.crop_height, 'width':width,
                                        'order':crop['order'], 'pred':crop['pred'], 'confidence':crop['confidence']})
    return paging, img_list, crop_imgs

def update_crop_images(request_data):
    collection = db['crop_images']
    collection.insert_one(request_data)
