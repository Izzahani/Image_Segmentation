# %% Import packages
import os, cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers,losses,callbacks
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import datetime
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split

# %% Step 1) Data Loading
# training data
TRAIN_PATH = os.path.join(os.getcwd(), 'dataset', 'train')

# testing data
TEST_PATH = os.path.join(os.getcwd(), 'dataset', 'test')

# %% 
# Load the images using opencv
# training data
images = []
masks = []

image_dir = os.path.join(TRAIN_PATH,'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images.append(img)
    
# Load the masks
mask_dir = os.path.join(TRAIN_PATH,'masks')
for mask_file in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)

# %%
# testing data
images_test = []
masks_test = []

image_dir = os.path.join(TEST_PATH,'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images_test.append(img)
    
# Load the masks
mask_dir = os.path.join(TEST_PATH,'masks')
for mask_file in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks_test.append(mask)

# %% Step 2) Data Pre-processing
# Convert the lists into numpy array
images_np = np.array(images)
masks_np = np.array(masks)
images_np_test = np.array(images_test)
masks_np_test = np.array(masks_test)

# %%
# Check some examples for train data
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(images_np[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(masks_np[i])
    plt.axis('off')
    
plt.show()

#%%
# Expand the mask dimension 
# training data
masks_np_exp = np.expand_dims(masks_np,axis=-1)
#Check the mask output
print(np.unique(masks[0]))

# testing data
masks_np_exp_test = np.expand_dims(masks_np_test,axis=-1)
#Check the mask output
print(np.unique(masks_test[0]))

#%%
# Convert the mask values into class labels
# training data
converted_masks = np.round(masks_np_exp/255).astype(np.int64)
#Check the mask output
print(np.unique(converted_masks[0]))

# testing data
converted_masks_test = np.round(masks_np_exp_test/255).astype(np.int64)
#Check the mask output
print(np.unique(converted_masks_test[0]))
#%%
# Normalize image pixels value
# training data
converted_images = images_np / 255.0
sample = converted_images[0]

# testing data
converted_images_test = images_np_test / 255.0
sample_test = converted_images_test[0]

#%%
# Perform train-val split
SEED = 12345
x_train,x_val,y_train,y_val = train_test_split(converted_images,converted_masks,test_size=0.2,random_state=SEED)

# %%
# Convert the numpy arrays into tensor 
x_train_tensor = tf.data.Dataset.from_tensor_slices(x_train)
x_val_tensor = tf.data.Dataset.from_tensor_slices(x_val)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_val_tensor = tf.data.Dataset.from_tensor_slices(y_val)
x_test_tensor = tf.data.Dataset.from_tensor_slices(converted_images_test)
y_test_tensor = tf.data.Dataset.from_tensor_slices(converted_masks_test)

# %%
# Combine the images and masks using zip
train_dataset = tf.data.Dataset.zip((x_train_tensor,y_train_tensor))
val_dataset = tf.data.Dataset.zip((x_val_tensor,y_val_tensor))
test_dataset = tf.data.Dataset.zip((x_test_tensor,y_test_tensor))

#%%
#[EXTRA] Create a subclass layer for data augmentation
class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal',seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
    
#%%
# Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

val_batches = val_dataset.batch(BATCH_SIZE)
test_batches = test_dataset.batch(BATCH_SIZE)

# %%
# Visualize some examples
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for images, masks in train_batches.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])

#%% Step 3) Model Development
# Create image segmentation model
# Use a pretrained model as the feature extraction layers
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

# List down some activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Define the feature extraction model
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    #Apply functional API to construct U-Net
    #Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    #This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

#%%
#Make of use of the function to construct the entire U-Net
OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
#Compile the model
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
keras.utils.plot_model(model, show_shapes=True)

#%% Step 4) Model Evaluation
#Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])
        

# %%
#Test out the show_prediction function
show_predictions()

#%%
#Create a callback to help display results during model training
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))

# %%
#Early stopping and tensorboard
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)
 
#%%
#Step 5) Model Training
#Hyperparameters for the model
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(val_dataset)//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(train_batches,validation_data=val_batches,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,callbacks=[DisplayCallback(),tensorboard_callback, early_stop_callback])

#%% Step 6) Model Deployment
show_predictions(test_batches,3)
print(model.evaluate(test_batches))


#%% Step 7) Model Saving
# save model
model.save('model.h5')
# %%
