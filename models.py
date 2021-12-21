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
    user_data = collection.find_one(filter={'email':request_data['email']})
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
        collection.insert_one(request_data)
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
    collection.update_one({'path':request_data['path']}, {'$set':request_data}, upsert=True)

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

    count = collection.count_documents({})
    paging = paginate(page, per_page, count)
    data_list = collection.find(sort=[('path', -1)]).limit(per_page).skip(offset)

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
    count = collection.count_documents({'path_folder':path_folder})
    paging = paginate(page, per_page, count)
    crop_list = collection.find({'path_folder':path_folder}, sort=[('order', 1)]).limit(per_page).skip(offset)

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

    count = collection.count_documents({})
    paging = paginate(page, per_page, count)
    data_list = collection.find(sort=[('path', -1)]).limit(per_page).skip(offset)

    collection = db['crop_images']
    crop_count = collection.count_documents({})
    target_count = collection.count_documents({'target':{'$exists':'true'}})
    total['count'] = crop_count
    total['target'] = target_count
    if count == 0:
        total['learning_rate'] = 0
    else:
        total['learning_rate'] = round(target_count / crop_count * 100, 2)

    img_list = []
    for data in data_list:
        path = data['path'] # results/20210425214433.jpg
        height = int(data['height'] / data['width'] * args.result_width)
        img_list.append({'path':path, 'name':path.split('/')[1], 'height':height, 'width':args.result_width})
        path_folder = path.split('.')[0]       # results/20210425214433
        static_path = args.static_folder + path_folder  # static/results/20210425214433
        crop_count = 0
        target_count = 0
        learning_rate = 0
        if os.path.isdir(static_path):
            crop_count = collection.count_documents({'path_folder':path_folder})
            target_count = collection.count_documents({'path_folder':path_folder , 'target':{'$exists':'true'}})
            if crop_count != 0:
                learning_rate = round(target_count / crop_count * 100, 1)
        img_list[-1]['count'] = crop_count
        img_list[-1]['target'] = target_count
        img_list[-1]['learning_rate'] = learning_rate
        if 'width_height_ratio' in data:
            img_list[-1]['width_height_ratio'] = data['width_height_ratio']
        img_list[-1]['confidence'] = data['avg_confidence']
        if 'model' in data:
            img_list[-1]['model'] = data['model']
    return paging, img_list, total

def avg_confidence():
    collection = db['images']
    data_list = collection.find(sort=[('path', -1)])

    for data in data_list:
        collection = db['crop_images']
        crop_list = collection.find({'path_folder':data['path'].split('.')[0]})
        count = collection.count_documents({'path_folder':data['path'].split('.')[0]})
        if count > 0:
            total_confidence = 0
            for crop in crop_list:
                total_confidence = total_confidence + crop['confidence']
            avg_confidence = round(total_confidence/count, 2)
        else:
            avg_confidence = 0

        request_data = {'path':data['path'], 'avg_confidence':avg_confidence}
        update_image(request_data)

def get_statistics():
    collection = db['crop_images']
    data_list = collection.find({'ed':{'$exists':'true'}})
    xy_list = []
    for data in data_list:
        xy_list.append({'confidence':data['confidence'], 'ed':data['ed']})
    collection = db['images']
    data_list = collection.find(sort=[('path', 1)])
    path_list = []
    confidence_list = []
    for data in data_list:
        path = data['path'].split('/')[1]
        path = path.split('.')[0]
        path_list.append(path)
        confidence_list.append(data['avg_confidence'])
    return xy_list, path_list, confidence_list

# train
def get_dataset():
    collection = db['crop_images']
    data_list = collection.find({'target':{'$exists':'true'}})
    img_list = []
    for data in data_list:
        img_list.append({'path':args.static_folder + data['path_folder'] + '/' + data['name'], 'label':data['target']})
    return img_list

def update_training_summary(request_data):
    collection = db['train']
    request_data['name'] = str(datetime.datetime.now().strftime('%Y%m%d%H%M')) + '.pth'
    collection.insert_one(request_data)
    return request_data['name']

def update_training_result(request_data):
    collection = db['train']
    collection.insert_one(request_data)

def get_loss():
    collection = db['train']
    per_page = 5
    summary_list = collection.find({'name':{'$exists':'true'}}, sort=[('name', -1)]).limit(per_page)
    loss_data = {}
    for summary in summary_list:
        data_list = collection.find({'model':summary['name']}, sort=[('epoch', 1)])
        loss_data[summary['name']] = []
        for data in data_list:
            loss_data[summary['name']].append(data)
    return loss_data


