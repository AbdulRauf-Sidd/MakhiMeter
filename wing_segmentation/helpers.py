from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from rembg import remove
from keras.utils import normalize
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras import backend as K
import tensorflow as tf



def preprocess_image(image_path, dimensions):
    with Image.open(image_path) as img:
        img = remove(img)
        img_gray = img.convert('L')
        img_resized = img_gray.resize(dimensions, Image.Resampling.LANCZOS)
        return img_resized
    

def process_image(image):
    test_img = np.array(image)
    test_img = normalize(test_img, axis=1)
    test_img = np.expand_dims(test_img, axis=-1)
    test_img_input = np.expand_dims(test_img, axis=-1)
    test_img_input= np.expand_dims(test_img_input, 0)
    return test_img_input

def predict(image):    
    model = multi_unet_model()
    model.load_weights('/home/abdulrauf/Projects/makhi_meter/wing_segmentation/models/unet_256_bg_removed_13000_checkpoint_epoch-21_val_loss-0.2400.keras')  
    prediction = (model.predict(image))
    return np.argmax(prediction, axis=3)[0,:,:]

def post_process(image, original_image, size):
    predicted_img = (image / image.max()) * 255  
    predicted_img = predicted_img.astype(np.uint8) 
    predicted_img = cv2.resize(predicted_img, size, interpolation=cv2.INTER_NEAREST)
    return label_and_save_areas(predicted_img, original_image)
    

def label_and_save_areas(inp, orginal_img):

    unique_values = np.unique(inp)

    names = {
        36: "2P",
        72: "3P",
        109: "S",
        145: "D",
        182: "M",
        218: "1P",
        255: "B1",
    }

    output_image = Image.fromarray(inp).convert('RGB')  
    draw = ImageDraw.Draw(output_image)
    areas = {'2P': 0, '3P': 0, 'S': 0, 'D': 0, 'M': 0, '1P': 0, 'B1': 0}

    for value in unique_values:
        if value == 0:
            continue
        
        
        mask = np.where(inp == value, 1, 0).astype(np.uint8)
        
        # Calculate area in pixels and convert to real-world units
        pixel_count = np.sum(mask)
        area = pixel_count * ((100/1450)**2)


        # Calculate centroid (center of mass) using image moments
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            if value in names:
                name = names[value]
                draw.text((cX, cY), f'{name}', fill=(211, 247, 5))  # Yellow text
                areas[name] = area
                
    return output_image, areas



#----------------------------------------------------------------------------------------------------------------------------------------------------
def multi_unet_model(n_classes=8, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model