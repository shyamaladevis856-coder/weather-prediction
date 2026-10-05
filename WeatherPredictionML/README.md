# Smart Weather Prediction using Machine Learning
Decision Tree Classifier + Flask website (Login, Register, Dashboard, History).

## Run in VS Code
1. Open this folder in VS Code (File > Open Folder).
2. Open terminal (Ctrl + `) and create a virtual environment:
   - Windows:  `python -m venv venv`  then  `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv`  then  `source venv/bin/activate`
3. Install libraries:  `pip install -r requirements.txt`
4. (Optional, already included) Re-create data + model:
   `python generate_dataset.py` then `python train_model.py`
5. Start the website:  `python app.py`
6. Open  http://127.0.0.1:5000  -> Register -> Login -> Dashboard.

## Use your own dataset
Replace `dataset/weather.csv` with columns `Temperature, Humidity, WindSpeed, Weather`,
then run `python train_model.py` again. The dashboard updates automatically.

## Folder structure
dataset/ (weather.csv) | model/ (trained model, metrics, tree rules) | templates/ | static/
app.py (website) | train_model.py (Decision Tree) | generate_dataset.py
