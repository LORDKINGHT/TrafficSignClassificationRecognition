import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Función para cargar y procesar las imágenes
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (32, 32))  # Ajusta el tamaño de la imagen
            images.append(img)
            labels.append(filename.split('_')[0])  # Suponiendo que el nombre del archivo es la etiqueta
    return np.array(images), np.array(labels)

# Ruta al directorio que contiene las imágenes de señales de tráfico
folder_path = "Todo"

# Cargar las imágenes y las etiquetas desde la carpeta
images, labels = load_images_from_folder(folder_path)

# Convertir etiquetas a números enteros
label_to_idx = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.array([label_to_idx[label] for label in labels])

# Particionamiento del conjunto de datos
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalización de los valores de píxeles de las imágenes
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Crear el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(48, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(43, activation='softmax')  # 43 clases para señales de tráfico
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=50, batch_size=1, validation_data=(test_images, test_labels))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# Predicciones en el conjunto de prueba
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Matriz de confusión
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Métricas de evaluación
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average='weighted')
recall = recall_score(test_labels, predicted_labels, average='weighted')
f1 = f1_score(test_labels, predicted_labels, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Graficar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_to_idx.keys(), yticklabels=label_to_idx.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
