import os
import re
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from PIL import Image, ImageFilter
import pytesseract

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'txt', 'pdf', 'docx'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# POS tagging function
def extract_pos_tags(text):
    tokens = word_tokenize(text)
    return pos_tag(tokens)

# NER function
def extract_named_entities(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    chunked = ne_chunk(pos_tags)
    named_entities = []

    # Regex pattern for dates
    date_pattern = r'\b\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
    date_matches = re.findall(date_pattern, text)
    for match in date_matches:
        named_entities.append((' '.join(match), 'DATE'))

    known_names = {'Mounika', 'Alice', 'Bob'}  # Add any specific names here

    for subtree in chunked:
        if isinstance(subtree, Tree):
            entity = " ".join([word for word, tag in subtree.leaves()])
            entity_type = subtree.label()
            if entity not in [e[0] for e in named_entities]:
                named_entities.append((entity, entity_type))
        else:
            word, tag = subtree
            if tag == 'NNP' and word[0].isupper() and word not in known_names:
                named_entities.append((word, 'PERSON'))

    return named_entities

# OCR function to extract text from an image
def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert('L')
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = image.resize((image.width * 2, image.height * 2))
        text = pytesseract.image_to_string(image, lang="eng")
        return text
    except Exception as e:
        print("Error during OCR processing:", e)
        return None

# Route for main page
@app.route("/", methods=["GET", "POST"])
def index():
    
    error_message = None
    success_message = None
    sentence = None
    pos_tags = None
    ner_result = None

    if request.method == "POST":
        option = request.form.get("option")

        if option == "own_text":
            # Get text input
            sentence = request.form.get("text_input")
            if sentence:
                pos_tags = extract_pos_tags(sentence)
                ner_result = extract_named_entities(sentence)
            else:
                error_message = "No text provided for analysis."

        elif option == "upload_image":
            # Handle image upload
            file = request.files.get("file")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Extract text using OCR
                sentence = extract_text_from_image(filepath)
                if sentence:
                    pos_tags = extract_pos_tags(sentence)
                    ner_result = extract_named_entities(sentence)
                else:
                    error_message = "Unable to extract text from the uploaded image."
            else:
                error_message = "Invalid file type or no file uploaded for image."

        elif option == "upload_document":
            # Handle document upload
            file = request.files.get("file")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Extract text if it's a .txt file
                if filename.endswith('.txt'):
                    with open(filepath, 'r') as f:
                        sentence = f.read()
                    pos_tags = extract_pos_tags(sentence)
                    ner_result = extract_named_entities(sentence)
                else:
                    error_message = "Unsupported document format. Please upload a .txt file."
            else:
                error_message = "Invalid file type or no file uploaded for document."

    return render_template("index.html", sentence=sentence, pos_tags=pos_tags, ner_result=ner_result, 
                           error_message=error_message, success_message=success_message)
# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # You can implement image processing and NER here

    return redirect(url_for('index'))
if __name__ == "__main__":
    app.run(debug=True)
