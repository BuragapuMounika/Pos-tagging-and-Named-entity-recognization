from flask import Flask, render_template, request
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import os
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import pdfplumber
import docx




# Initialize Flask app
app = Flask(__name__)

# Set up file upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# NER model pipeline from Hugging Face
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to perform POS tagging and NER
def pos_ner(text):
    # Perform NER using Hugging Face pipeline
    ner_results = ner_pipeline(text)

    # Format NER results to show entity and its type
    ner_result_formatted = []
    for result in ner_results:
        word = result['word']
        entity_type = result.get('entity_group', 'N/A')
        ner_result_formatted.append((word, entity_type))

    # Perform POS tagging using NLTK
    tokens = word_tokenize(text)
    pos_results = pos_tag(tokens)

    return pos_results, ner_result_formatted

# Function to extract text from uploaded files
def extract_text_from_file(filepath, file_extension):
    text = ""
    if file_extension == "txt":
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    elif file_extension == "pdf":
        with pdfplumber.open(filepath) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages])
    elif file_extension == "docx":
        doc = docx.Document(filepath)
        text = ' '.join([para.text for para in doc.paragraphs])
    elif file_extension in ["png", "jpg", "jpeg"]:
        image = Image.open(filepath)
        text = pytesseract.image_to_string(image)
    return text

# Route to handle the home page and form submission
@app.route("/", methods=["GET", "POST"])
def index():
    pos_tags = None
    ner_result = None
    sentence = None
    error_message = None

    if request.method == "POST":
        option = request.form.get("option")  # Retrieve the option selected by the user

        # Handle user input text
        if option == "own_text":
            sentence = request.form.get("text_input")
            if sentence:
                pos_tags, ner_result = pos_ner(sentence)
            else:
                error_message = "Please enter some text."

        # Handle file upload (documents or images)
        elif option in ["upload_image", "upload_document"]:
            file = request.files.get('file')  # <-- Ensure the file is retrieved correctly
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Get file extension
                file_extension = filename.rsplit('.', 1)[1].lower()

                # Extract text from the uploaded file
                sentence = extract_text_from_file(filepath, file_extension)

                if sentence:
                    pos_tags, ner_result = pos_ner(sentence)
                else:
                    error_message = "Failed to extract text from the uploaded file."
            else:
                error_message = "File type not allowed or no file uploaded."

    return render_template("index.html", sentence=sentence, pos_tags=pos_tags, ner_result=ner_result, error_message=error_message)

# Run the Flask app
if __name__ == "__main__":  # Corrected "__omain__" to "__main__"
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

    