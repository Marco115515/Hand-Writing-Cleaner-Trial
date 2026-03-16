from tensorflow import keras
import numpy as np
import cv2

#########################

input_path = 'sample_2.jpg'                       # Input path here
output_path = 'sample_2_cleaned.jpg'              # Output path here

#########################

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)

def predict_mask(img):
    x, y = img.shape
    padded_x, padded_y = int(512+max(np.ceil((x-512)/384)*384, 0)), \
                            int(512+max(np.ceil((y-512)/384)*384, 0))
    padded_img = np.pad(img, ((0, padded_x-x), (0, padded_y-y)))/255
    mask = np.zeros((padded_x, padded_y))
    for i in range(0, max(0, x-512)+1, 384):
        for j in range(0, max(0, y-512)+1, 384):
            predict = model.predict(padded_img[i:i+512, j:j+512].reshape((1, 512, 512, 1)), verbose=0)
            predict = predict.reshape((512, 512))
            mask[i:i+512, j:j+512] = np.maximum(mask[i:i+512, j:j+512], predict)
    mask = np.where(mask<0.5, 0, 1)
    return mask[:x, :y]

custom_objects = {"dice_loss": dice_loss}
model = keras.models.load_model("model_1.0.keras", custom_objects=custom_objects)

img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = predict_mask(img)
mask255 = (mask*255).astype(np.uint8)
masked = np.where(mask255>img, mask255, img).astype(np.uint8)
cv2.imwrite(output_path, masked)