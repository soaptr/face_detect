from app import app
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = 'app/static/images/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    image = request.args.get('image')
    if not image:
        image='images/example.bmp'
    return render_template('index.html', image=image)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(app.config['UPLOAD_FOLDER'] + filename)
        return redirect(url_for('index', image='images/'+filename))
        #return render_template('index.html', image='images/'+filename)
    return redirect(url_for('index'))