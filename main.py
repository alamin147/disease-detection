import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

st.header("Disease Recognition")
test_image = st.file_uploader("Choose an Image:")
if(st.button("Show Image")):
    st.image(test_image,width=4,use_column_width=True)
#Predict button
if(st.button("Predict")):
    st.write("Our Prediction")
    result_index = model_prediction(test_image)
    #Reading Labels
    class_name = ['Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
    st.success("Model is Predicting it's a {}".format(class_name[result_index].split("___")[1]))