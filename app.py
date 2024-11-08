import streamlit as st
import numpy as np
from keras.models import load_model # type: ignore
from keras.preprocessing.image import load_img, img_to_array # type: ignore


# Load the pre-trained model
model_path = r'C:\Users\DELL\Desktop\new\flower_classification_model.h5'
model = load_model(model_path)

# Define the class labels including non-flowers
class_labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

def preprocess_image(image):
    # Resize the image to the input size of the model
    image = load_img(image, target_size=(150, 150))  # Adjust size as per your model's requirement
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize the image
    return image

# Streamlit app layout
st.title("Flower Classification App")
st.write("Upload an image of a flower or non-flower to classify it.")

# File uploader for flower images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the uploaded image
    processed_image = preprocess_image(uploaded_file)

    # Make predictions
    predictions = model.predict(processed_image)

    # Check if predictions are made successfully
    if predictions is not None and len(predictions) > 0:
        # Get predicted class index and confidence score
        predicted_class_index = np.argmax(predictions[0])
        confidence_score = predictions[0][predicted_class_index]  # Ensure this line is present

        # Debugging output for predictions
        st.write(f"Predicted Class Index: {predicted_class_index}")
        st.write(f"Confidence Scores: {predictions[0]}")

        # Set a confidence threshold
        confidence_threshold = 0.6

        if confidence_score < confidence_threshold:
            predicted_class_label = "Not a Flower"
        else:
            predicted_class_label = class_labels[predicted_class_index]

        # Display prediction results
        st.write(f"Prediction: {predicted_class_label} with a confidence of {confidence_score:.2f}")
    else:
        st.write("Prediction failed. Please check your model and input data.")