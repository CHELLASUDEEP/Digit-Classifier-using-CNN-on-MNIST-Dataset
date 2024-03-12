# Digit Classifier with CNNs on MNIST Dataset: Deployment and Feedback Loop

NOTE:- USE VISUAL STUDIO CODE FOR EASY EXECUTION

A Streamlit web application for categorising handwritten digits using a convolutional neural network (CNN) model that has been pre-trained on the MNIST dataset. Users of the application can upload photographs for categorization, leave comments, and start real-time fine-tuning. The dynamically updated model (`model_fine.h5`) the initial pre-trained model (`model_1.h5`), and a system-scheduled job runs periodically to refine the model depending on user feedback.

# File Contents

1) app.py :- The code of the Streamlit web application, which offers options for instantaneous model fine-tuning, user feedback, and image classification.
2) fine_tune.py :- File containing functions for scheduled fine-tuning of the pre-trained model based on user feedback.
3) CNN_model.ipynb :- In this code we have trained the MNIST dataset and saved the pre trained model(model_1.h5) for user feedback and knowing about the incorrect predictions.
4) model_1.h5 :- Pretrained Model.
5) train.csv :- CSV file containing training data for model evaluation during fine-tuning.
6) incorrect_predictions :- Folder to store images that resulted in incorrect predictions.
7) model_fine.h5 :- Model after fine tuning
8) test images :- Use these sample images for finding any incorrect predictions

# Instructions

1) Environment Setup :- Make sure that all the required modules and libraries mentioned in the requirements.txt file are installed.

2) Model Building :- Explore the ipynb file `CNN_model.ipynb` to understand the process of building the initial pre-trained model (`model_1.h5`). Execute the notebook if you wish to retrain or modify the model architecture.

3) Execute app.py :- In app.py code at 8th line, mention the path at which you have saved the pre trained model(`model_1.h5`) acoording to your requirement. Now execute the code in the terminal using the command:

            streamlit run app.py

4) Usage :- After running the app.py file in your web browser, you will be directed to a URL where streamlit app is present. First Upload the image (Present in test images). After uploading : 
    **i** Model prediction will be printed.
    **ii** Now, a button will appear.
    **iii** If the prediction matches with image, no need to do anything.
    **iv** If not matches, then select incorrect button. a input line appears and enter the correct value in it
    **v** After entering, image will be saved in the incorrect_predictions folder.

5) Execute fine_tune.py :- In fine_tune.py code, mention the path of train.csv dataset in line 75 and mention the path of the pretrained model (`model_1.h5`) in line 77 and mention the path of incorrect_predictions folder in line 80. Now execute the code.
                fine tuned model (`model_fine.h5`) will be saved

6) Now execute the app.py code one more time. But, in line 8 take the path of the fine tuned model (`model_fine.h5`). Execute the code. Streamlit will appear on your web browser. Take the image which has given the wrong prediction. It gives a correct predicted value now.