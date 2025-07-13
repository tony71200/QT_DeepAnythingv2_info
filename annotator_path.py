import os

models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')

models_path = os.path.realpath(models_path)
os.makedirs(models_path, exist_ok=True)
