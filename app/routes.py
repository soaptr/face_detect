from app import app
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import gc
from utils import utils


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'app/static/images/'
app.config['EXAMPLE_IMAGE'] = 'example.jpg'
app.config['CHECKPOINT_PATH'] = 'app/static/yolov5.pt'
model = utils.load_model(app.config['CHECKPOINT_PATH'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    image = request.args.get('image')
    boxes = request.args.get('boxes')
    if not image:
        image = app.config['EXAMPLE_IMAGE']
    if not boxes:
        boxes = 10
    boxes = int(boxes)
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if (not (image in f)) and (not (app.config['EXAMPLE_IMAGE'] in f)):
            os.remove(app.config['UPLOAD_FOLDER']+f)
    return render_template('index.html', image=image, boxes=boxes)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = app.config['UPLOAD_FOLDER'] + filename
        file.save(img_path)
        img = utils.load_image(img_path)
        n_boxes = utils.find_faces(model, img, filename, app.config)
        del img, file
        gc.collect()
        return redirect(url_for('index', image=filename, boxes=n_boxes))
    return redirect(url_for('index'))