# KiteDetector
A streamlit application whereby a trained YoloV8 object detection model will take an input video and detect the kites within it.

# Instructions on running the code locally
Install all dependencies using `pip install -r requirements.txt` 

# How to run locally
You can run this locally by using `streamlit run app.py`. Make sure you are using the shell in the project root directory. If you want to make predictions on your own data locally (much better results) then you can add your video in the `testData` folder, adjust the code of `predict.py` according to the path of your video and run that script in regular Python fashion. 
