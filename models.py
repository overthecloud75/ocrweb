import os
from werkzeug.security import check_password_hash

import datetime
from nltk.metrics.distance import edit_distance

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

def update_crop_image(request_data):
    collection = db['crop_images']
    if 'target' in request_data:
        data = collection.find_one({'path_folder':request_data['path_folder'], 'order':request_data['order']})
        # ICDAR2019 Normalized Edit Distance
        pred = data['pred']
        gt = request_data['target']
        if len(gt) == 0 or len(pred) == 0:
            norm_ED = 0
        elif len(gt) > len(pred):
            norm_ED = 1 - edit_distance(pred, gt) / len(gt)
        else:
            norm_ED = 1 - edit_distance(pred, gt) / len(pred)
        request_data['ed'] = round(norm_ED, 3)
    collection.update_one({'path_folder':request_data['path_folder'], 'order':request_data['order']}, {'$set':request_data}, upsert=True)

def get_images(page=1):
    collection = db['images']

    per_page = 5
    offset = (page - 1) * per_page

    data_list = collection.find(sort=[('path', -1)])
    count = data_list.count()
    paging = paginate(page, per_page, count)
    data_list = data_list.limit(per_page).skip(offset)

    collection = db['crop_images']
    img_list = []

    crop_imgs = {}
    for data in data_list:
        path = data['path'] # results/20210425214433.jpg
        height = int(data['height'] / data['width'] * args.result_width)
        img_list.append({'path':path, 'name':path.split('/')[1], 'height':height, 'width':args.result_width})
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

def get_detail(page=1, filename=None):
    collection = db['crop_images']

    per_page = page_default['per_page']
    offset = (page - 1) * per_page

    path_folder = args.result_folder + filename.split('.')[0]
    crop_list = collection.find({'path_folder':path_folder}, sort=[('order', 1)])
    count = crop_list.count()
    paging = paginate(page, per_page, count)
    crop_list = crop_list.limit(per_page).skip(offset)

    crop_imgs = []
    for crop in crop_list:
        width = int(crop['width'] / crop['height'] * args.crop_height)
        crop_imgs.append({'path': path_folder + '/' + crop['name'], 'height': args.crop_height, 'width': width,
                          'order': crop['order'], 'pred': crop['pred'], 'confidence': crop['confidence']})
        if 'target' in crop:
            crop_imgs[-1]['target'] = crop['target']
        if 'ed' in crop:
            crop_imgs[-1]['ed'] = crop['ed']
    return paging, crop_imgs

def get_summary(page=1):
    collection = db['images']
    total = {}

    per_page = page_default['per_page']
    offset = (page - 1) * per_page

    data_list = collection.find(sort=[('path', -1)])
    count = data_list.count()
    paging = paginate(page, per_page, count)
    data_list = data_list.limit(per_page).skip(offset)

    collection = db['crop_images']
    crop_count = collection.find().count()
    target_count = collection.find({'target':{'$exists':'true'}}).count()
    total['count'] = crop_count
    total['target'] = target_count
    if count == 0:
        total['learning_rate'] = 0
    else:
        total['learning_rate'] = round(target_count / crop_count * 100, 2)

    img_list = []
    for data in data_list:
        path = data['path'] # results/20210425214433.jpg
        model = None
        if 'model' in data:
            model = data['model']
        height = int(data['height'] / data['width'] * args.result_width)
        img_list.append({'path':path, 'name':path.split('/')[1], 'height':height, 'width':args.result_width})
        path_folder = path.split('.')[0]       # results/20210425214433
        static_path = args.static_folder + path_folder  # static/results/20210425214433
        crop_count = 0
        target_count = 0
        learning_rate = 0
        if os.path.isdir(static_path):
            crop_list = collection.find({'path_folder':path_folder}, sort=[('order', 1)])
            crop_count = crop_list.count()
            crop_list = collection.find({'path_folder':path_folder , 'target':{'$exists':'true'}}, sort=[('order', 1)])
            target_count = crop_list.count()
            if crop_count != 0:
                learning_rate = round(target_count / crop_count * 100, 1)
        img_list[-1]['count'] = crop_count
        img_list[-1]['target'] = target_count
        img_list[-1]['learning_rate'] = learning_rate
        img_list[-1]['model'] = model
    return paging, img_list, total

def get_graph():
    collection = db['crop_images']
    data_list = collection.find({'ed':{'$exists':'true'}})
    xy_list = []
    for data in data_list:
        xy_list.append({'confidence':data['confidence'], 'ed':data['ed']})
    return xy_list

# train
def get_dataset():
    collection = db['crop_images']
    data_list = collection.find({'target':{'$exists':'true'}})
    img_list = []
    for data in data_list:
        img_list.append({'path':args.static_folder + data['path_folder'] + '/' + data['name'], 'label':data['target']})
    return img_list

