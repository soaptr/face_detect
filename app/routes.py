from app import app
from flask import render_template, request, redirect, url_for

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

#@app.route('/upload', methods=['POST'])
#def upload_file():
#    file = request.files['file']
#    if file and allowed_file(file.filename):
#        filename = secure_filename(file.filename)
#        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#        return redirect(url_for('uploaded_file',
#                                filename=filename))
#    return redirect('/')