# !pip install gradio ipywidgets
import joblib
import tensorflow as tf
import gradio as gr
import numpy as np

model = tf.keras.models.load_model("model.keras")
classes = joblib.load("classes.joblib")

def predict(path):
    image = tf.keras.preprocessing.image.load_img(path, target_size=(150, 150))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    predicted = model.predict(image)[0].argmax(axis=-1)
    return classes[predicted]

# https://www.gradio.app/guides
with gr.Blocks() as blocks:
    path = gr.Image(label="Image", type="filepath")
    label = gr.Textbox(label="Label")

    inputs = [path]
    outputs = [label]

    predict_btn = gr.Button("Predict")
    predict_btn.click(predict, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    blocks.launch() # Local machine only
    # blocks.launch(server_name="0.0.0.0") # LAN access to local machine
    # blocks.launch(share=True) # Public access to local machine
    # predict("cats_vs_dogs/cat/0.jpg")
