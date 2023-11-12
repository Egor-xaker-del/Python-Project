import cv2
import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from statistics import mode

# Загрузка предварительно обученной модели MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False)

class_names = [
    "cartwheel", "catch", "clap", "climb", "dive", "draw_sword",
    "dribble", "fencing", "flic_flac", "golf", "handstand", "hit",
    "jump", "pick", "pour", "pullup", "push", "pushup", "shoot_ball",
    "sit", "situp", "swing_baseball", "sword_exercise", "throw"
]

# Создаем словарь для соответствия индекса класса его названию
class_dict = {i: class_name for i, class_name in enumerate(class_names)}

# Задайте количество классов в вашем наборе данных
num_classes = 10  # Замените на фактическое количество классов

# Создайте кастомную модель поверх MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Подготовьте видео и выполните предсказания
video_path = './video.avi'

cap = cv2.VideoCapture(video_path)

predicted_classes = []  # Создаем массив для хранения предсказанных классов

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

    # Добавляем предсказанный класс в массив
    predicted_classes.append(predicted_class)

    # Отображение кадра с выводом результата (настройте под свои нужды)
    #cv2.imshow('Video', frame[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# В predicted_classes теперь хранятся предсказанные классы
most_common_class = mode(predicted_classes)

print("Номер дейвствия на видео:", most_common_class)
print("Название действия:", class_dict[most_common_class])
