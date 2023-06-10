import os

from flask import Flask, render_template_string, jsonify, send_from_directory
import auxfunctions as aux

app = Flask(__name__)

from flask import Flask, render_template_string, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename

from flask import Flask, render_template_string, request, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # replace with your secret key
app.config['UPLOAD_PATH'] = os.path.join('uploads')  # replace with your path
Bootstrap(app)

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileRequired()])

@app.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.photo.data
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_PATH'] + '/' + filename)
        return render_template_string('''
            <!doctype html>
            <title>Upload new File</title>
            <h1>File uploaded successfully</h1>
            <img src="{{ url_for('get_image', filename=filename) }}" alt="Image">
            <a href="{{ url_for('upload') }}">Upload another file</a>
        ''', filename=filename)
    return render_template_string('''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method="POST" enctype="multipart/form-data">
            {{ form.hidden_tag() }}
            <p>
                {{ form.photo.label }}<br>
                {{ form.photo() }}
            </p>
            <p><input type="submit" value="Upload"></p>
        </form>
    ''', form=form)

@app.route('/uploads/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

if __name__ == '__main__':
    app.run(port=5000)

def my_function():
    return {"message": "Hello, World!"}


@app.route('/execute_my_function', methods=['GET'])
def execute_my_function():
    result = my_function()  # replace this with your function
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(port=5000)
