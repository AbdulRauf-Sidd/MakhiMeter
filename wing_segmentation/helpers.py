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
    
    # Open the image file
    with Image.open(image_path) as img:
        # Convert image to grayscale
        img = remove(img)
        img_gray = img.convert('L')

        # Resize the image
        img_resized = img_gray.resize(dimensions, Image.Resampling.LANCZOS)
        # Return the processed image
        return img_resized
    

def process_image(image):
    # test_img = np.array(image)
    # test_img = np.expand_dims(test_img, axis=2)
    # test_img = normalize(test_img, axis=1)
    # test_img_norm=test_img[:,:,0][:,:,None]
    # return np.expand_dims(test_img_norm, 0)
    test_img = np.array(image)
    test_img = normalize(test_img, axis=1)
    test_img = np.expand_dims(test_img, axis=-1)
    test_img_input = np.expand_dims(test_img, axis=-1)
    test_img_input=np.expand_dims(test_img_input, 0)
    return test_img_input

def predict(image):
    # model = unet(n_class=8)
    model = multi_unet_model()
    # model.load_weights('/home/abdulrauf/Projects/makhi_meter/wing_segmentation/models/unet_256_interpolated_6900_10epochs.h5')
    
    model.load_weights('wing_segmentation/models/unet_256_bg_removed_9600_checkpoint_epoch-19_val_loss-0.1870.keras')  
    prediction = (model.predict(image))
    return np.argmax(prediction, axis=3)[0,:,:]

def post_process(image, path, size, output_path):
    predicted_img = (image / image.max()) * 255  # Scale the prediction to 0-255
    predicted_img = predicted_img.astype(np.uint8)  # Convert to uint8
    predicted_img = cv2.resize(predicted_img, size, interpolation=cv2.INTER_NEAREST)
    area_per_pixel = (100/420)**2
    return label_and_save_areas(predicted_img, area_per_pixel, output_path)
    # Image.fromarray(labeled_image).save(path)
    

def label_and_save_areas(inp, area_per_pixel, output_text_file):
    # Load the grayscale image
    # image = cv2.imread(inp_path, 0)
    unique_values = np.unique(inp)

    # Define segment names corresponding to pixel values
    names = {
        36: "2P",
        72: "C",
        109: "M",
        145: "S",
        182: "D",
        218: "1P",
        255: "B1",
    }

    # Prepare to overlay text on the image
    output_image = Image.fromarray(inp).convert('RGB')  # Convert to RGB for text overlay
    draw = ImageDraw.Draw(output_image)
    areas = {'2P': 0, 'C': 0, 'M': 0, 'S': 0, 'D': 0, '1P': 0, 'B1': 0}
    # Open the text file for saving areas
    for value in unique_values:
        if value == 0:
            continue  # Skip background (assuming 0 is background)
        
        # Create a mask for this specific segment (all pixels with the same value)
        mask = np.where(inp == value, 1, 0).astype(np.uint8)
     # Calculate area in pixels and convert to real-world units
        pixel_count = np.sum(mask)
        area = pixel_count * area_per_pixel  # Area in micrometers squared
        # Calculate centroid (center of mass) using image moments
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print('hi3')
            # Add the name and area to the overlay if the value is in the dictionary
            if value in names:
                name = names[value]
                draw.text((cX, cY), f'{name}', fill=(211, 247, 5))  # Yellow text
             # Write the name and area to the text file
                areas[name] = area
                
    return output_image, areas

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)



def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)







def unet(pretrained_weights = None,input_size=(256, 256, 1), n_class=8):
    inputs = tf.keras.Input(shape=input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)
    

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([conv3,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv2,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(n_class, 1, activation = 'softmax')(conv8)

    model = tf.keras.Model(inputs = inputs, outputs = conv9)

    # model.compile(optimizer = Adam(lr = 0.0001), loss = ['binary_crossentropy'], metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 0.0001), loss = [dice_coef_loss], metrics = [dice_coef])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model




################################################################
def multi_unet_model(n_classes=8, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
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
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model