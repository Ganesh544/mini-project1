import os

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_directories():
    """Create necessary directories."""
    directories = ['models', 'results', 'uploads']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
