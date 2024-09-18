import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Create CNN Model for Currency Recognition
def create_currency_recognition_model(input_shape, num_classes):
    model = Sequential()
    
    # First Convolutional Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Third Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to avoid overfitting
    
    # Output Layer (for different currency denominations)
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Function to train model and check accuracy
def train_and_evaluate_model(model, train_generator, val_generator, model_name):
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=[early_stopping]
    )

    # Evaluate the model on validation set
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Validation Accuracy for {model_name}: {val_accuracy * 100:.2f}%")

    # Save the model
    model.save(f'{model_name}.h5')
    print(f"Model saved as {model_name}.h5")

    return model, val_accuracy * 100

# Preprocess the dataset using ImageDataGenerator
def preprocess_dataset(train_dir, val_dir, img_size=(128, 128), batch_size=32):
    # Create ImageDataGenerator for training and validation
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'  # Use 'categorical' for multi-class classification
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator

# Example usage
train_dir = r'C:\Users\Melissa\AppData\Local\Programs\Python\Python311\fakeDetect\preprocessing'  # Path to the training dataset
val_dir = r'C:\Users\Melissa\AppData\Local\Programs\Python\Python311\fakeDetect\preprocessing'  # Path to the validation dataset

# Preprocess the dataset
train_generator, val_generator = preprocess_dataset(train_dir, val_dir, img_size=(128, 128), batch_size=32)

# Get the number of classes (currency denominations)
num_classes = train_generator.num_classes

# Create the model
currency_model = create_currency_recognition_model(input_shape=(128, 128, 3), num_classes=num_classes)

# Train and evaluate the model
currency_model, currency_accuracy = train_and_evaluate_model(
    currency_model, 
    train_generator, 
    val_generator, 
    model_name='currency_detection_modelnew'
)

# Output the accuracy
print(f"Currency detection Model Accuracy: {currency_accuracy:.2f}%")
