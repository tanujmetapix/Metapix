import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the MirNet model
mirnet_model = tf.lite.Interpreter(model_path='lite-model_mirnet-fixed_fp16_1.tflite')
mirnet_model.allocate_tensors()

# Capture frames from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Preprocess the image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((400, 400), Image.ANTIALIAS)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Run the MirNet model on the preprocessed image
    mirnet_model.set_tensor(mirnet_model.get_input_details()[0]['index'], image)
    mirnet_model.invoke()
    output_image = mirnet_model.get_tensor(mirnet_model.get_output_details()[0]['index'])

    # Convert the output image to a format compatible with OpenCV
    output_image = np.squeeze(output_image, axis=0)
    output_image = output_image.clip(0, 1) * 255
    output_image = output_image.astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Display the corrected image
    cv2.imshow('Corrected Image', output_image)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

