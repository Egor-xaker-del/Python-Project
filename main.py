import cv2
import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
#from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense

# Загрузка предварительно обученной модели MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False)


# Задайте количество классов в вашем наборе данных
classes = {0: 'golf',1: 'pick' ,2: 'climb'}
num_classes = 3

# Создайте кастомную модель поверх MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Подготовьте видео и выполните предсказания
video_path = "./top.avi"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (224, 224))

    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)

    predictions = model.predict(frame)

    # Получите индекс класса с наибольшей вероятностью
    predicted_class = np.argmax(predictions)
    resolve = classes[predicted_class]
    # Выведите результат

    print(f"Predicted action: Class {resolve}")
    # Отображение кадра с выводом результата (настройте под свои нужды)
    cv2.imshow('Video', frame[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()