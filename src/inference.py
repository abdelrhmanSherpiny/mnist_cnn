from PIL import Image
import numpy as np
from io import BytesIO
from .utils.config import CLASS_NAMES, MODEL

def classify_image(image_bytes: bytes):
    try:
        # Open image from bytes
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'L':
            img = img.convert('L')
        
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0

        # Reshape for model input
        img_array = np.expand_dims(img_array, axis=(0, -1))

        prediction = MODEL.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction, axis=-1)[0]
        predicted_name = CLASS_NAMES[predicted_class]

        return {
            'class_index': int(predicted_class),
            'class_name': predicted_name,
            'confidence': float(prediction[0][predicted_class] * 100)
        }
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")
    