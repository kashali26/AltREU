import runpy
########################################################################################
########################################################################################
# Set Up the AI ( Get the information from dataset + create the all_images folder)
#runpy.run_path('/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/data_sort.py')
########################################################################################
########################################################################################
#Set Up the train folder with 50,000 xray images of patients with no findings and Pneumonia
# runpy.run_path('/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/train_copy_image.py')
########################################################################################
########################################################################################
#Set Up the val folder with 12,500 xray images of patients with no findings and Pneumonia
# runpy.run_path('/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/val_copy_image.py')
########################################################################################
########################################################################################
#Check to see if the Training and Validation datasets had any overlapping xray images
# import os

# # Define directories
# train_pneumonia_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/train/pneumonia'
# train_normal_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/train/normal'
# val_pneumonia_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/val/pneumonia'
# val_normal_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/val/normal'

# # Get lists of image files
# train_pneumonia_images = set(os.listdir(train_pneumonia_dir))
# train_normal_images = set(os.listdir(train_normal_dir))
# val_pneumonia_images = set(os.listdir(val_pneumonia_dir))
# val_normal_images = set(os.listdir(val_normal_dir))

# # Find overlaps
# overlap_train_val_pneumonia = train_pneumonia_images.intersection(val_pneumonia_images)
# overlap_train_val_normal = train_normal_images.intersection(val_normal_images)

# # Count the number of overlaps
# count_overlap_train_val_pneumonia = len(overlap_train_val_pneumonia)
# count_overlap_train_val_normal = len(overlap_train_val_normal)

# # Print counts
# print(f"Number of overlapping pneumonia images between train and val sets: {count_overlap_train_val_pneumonia}")
# print(f"Number of overlapping normal images between train and val sets: {count_overlap_train_val_normal}")
########################################################################################
########################################################################################
#Set Up the test folder with 6,250 xray images of patients with no findings and Pneumonia
# runpy.run_path('/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/test_copy_image.py')
########################################################################################
########################################################################################
#Check for overlap of xray in all three folders , test, train and val
# import os

# # Define directories for the datasets
# train_pneumonia_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/train/pneumonia'
# train_normal_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/train/normal'
# val_pneumonia_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/val/pneumonia'
# val_normal_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/val/normal'
# test_pneumonia_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/test/pneumonia'
# test_normal_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/test/normal'

# # Function to get a set of all file paths in a directory
# def get_file_set(src_dir):
#     return set(os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)))

# # Get sets of file paths for each category and folder
# train_pneumonia_files = get_file_set(train_pneumonia_dir)
# train_normal_files = get_file_set(train_normal_dir)
# val_pneumonia_files = get_file_set(val_pneumonia_dir)
# val_normal_files = get_file_set(val_normal_dir)
# test_pneumonia_files = get_file_set(test_pneumonia_dir)
# test_normal_files = get_file_set(test_normal_dir)

# # Check for overlaps
# def check_overlaps(set1, set2, name1, name2):
#     overlap = set1.intersection(set2)
#     if overlap:
#         print(f"Overlap found between {name1} and {name2}:")
#         for file in overlap:
#             print(f"  {file}")

# # Compare each folder
# check_overlaps(train_pneumonia_files, val_pneumonia_files, 'Train Pneumonia', 'Val Pneumonia')
# check_overlaps(train_pneumonia_files, test_pneumonia_files, 'Train Pneumonia', 'Test Pneumonia')
# check_overlaps(val_pneumonia_files, test_pneumonia_files, 'Val Pneumonia', 'Test Pneumonia')

# check_overlaps(train_normal_files, val_normal_files, 'Train Normal', 'Val Normal')
# check_overlaps(train_normal_files, test_normal_files, 'Train Normal', 'Test Normal')
# check_overlaps(val_normal_files, test_normal_files, 'Val Normal', 'Test Normal')

# # Check overlap between pneumonia and normal in each folder
# def check_pneumonia_vs_normal(pneumonia_files, normal_files, folder_name):
#     overlap = pneumonia_files.intersection(normal_files)
#     if overlap:
#         print(f"Overlap found within {folder_name}:")
#         for file in overlap:
#             print(f"  {file}")

# check_pneumonia_vs_normal(train_pneumonia_files, train_normal_files, 'Train')
# check_pneumonia_vs_normal(val_pneumonia_files, val_normal_files, 'Val')
# check_pneumonia_vs_normal(test_pneumonia_files, test_normal_files, 'Test')
########################################################################################
########################################################################################
#Creating a ResNet Model to help determine whether someone had pnemonia or not


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt

# # Directories
# train_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/train'
# val_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/val'
# test_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/test'

# # ImageDataGenerator for training, validation, and testing
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# val_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# # Load data from directories
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary'
# )
# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary'
# )
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary',
#     shuffle=False
# )

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=20,
#     validation_data=val_generator,
#     validation_steps=val_generator.samples // val_generator.batch_size,
#     callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
# )

# test_loss, test_accuracy = model.evaluate(
#     test_generator,
#     steps=test_generator.samples // test_generator.batch_size
# )
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()





import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Directories
 train_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/train'
 val_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/val'
 test_dir = '/Users/kashan/Desktop/Altreu AI Lung X-ray/Scripts/AI Images/test'

# ImageDataGenerator for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load VGG16 model pre-trained on ImageNet data
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Add custom layers on top
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(
    test_generator,
    steps=test_generator.samples // test_generator.batch_size
)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
