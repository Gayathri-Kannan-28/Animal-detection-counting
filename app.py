import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import csv
import os
from datetime import datetime
import base64
import pandas as pd
from collections import defaultdict
from PIL import Image


# Function to predict the species and accuracy
def predict_species(img):
    model = ResNet50(weights='imagenet')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    decoded_preds = decode_predictions(preds, top=1)[0]
    species_names = [pred[1].replace('_', ' ') for pred in decoded_preds]
    accuracies = [pred[2] for pred in decoded_preds]

    return species_names[0], accuracies[0]

# Function to save the results in a CSV file with the current date and time
def save_results_to_csv(results):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H-%M-%S")  # Remove colons from the timestamp
    file_name = "results.csv"
    species_counts = defaultdict(int)  # To count occurrences of each species
    # Read existing data to update counts
    try:
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                species_counts[row[2]] += 1
    except FileNotFoundError:
        pass

    with open(file_name, "a+", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["Date", "Time", "Species Name", "Accuracy", "Count"])
        for result in results:
            species_name = result[2]
            accuracy = result[3]
            count = species_counts[species_name]+1
            writer.writerow([result[0], result[1], species_name, accuracy, count])
            species_counts[species_name] += 1
    return file_name

# Streamlit app
def display_previous_results():
    try:
        df = pd.read_csv("results.csv")
        st.write("Previous Results:")
        st.write(df)
    except FileNotFoundError:
        st.warning("No previous results found.")

if "results" in st.session_state:
    with open(st.session_state.results, "rb") as f:
        data = f.read()
    st.sidebar.download_button("Download CSV", data, file_name=os.path.basename(st.session_state.results), key="download_button")

# Streamlit app
def main():
    st.title("Animal Species Prediction")
    st.title("Welcome!!!")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((150, 150))  # Resize image to 150x150 pixels
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        if st.button("Predict"):
            img = img.resize((224, 224))  # Resize image back to 224x224 pixels for model prediction
            img_array = image.img_to_array(img)
            species_name, accuracy = predict_species(img_array)
            st.write(f"Species: {species_name}")
            st.write(f"Accuracy: {accuracy}")

            # Save the results with the current date and time
            results = [(datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S"), species_name, accuracy)]
            file_name = save_results_to_csv(results)
            st.success(f"Results saved to {file_name}")

            # Store the CSV file path in session state
            st.session_state.results = file_name

    st.sidebar.button("Display Previous Results", on_click=display_previous_results)

if __name__ == "__main__":
    main()
