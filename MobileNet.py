# pip install googletrans
# pip install colorama

import tensorflow as tf
import matplotlib.pyplot as plt
from googletrans import Translator
import tkinter as tk
from tkinter import filedialog
from colorama import Fore, Back, Style


translator = Translator()

def image_to_tensor(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3)

    tensor = tf.convert_to_tensor(image)
    return tensor

def load_and_prepare_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename()
    file_label.config(text=file_path)
    file_label.quit()

def select_image_file():
    global file_label
    root = tk.Tk()
    root.geometry("600x400")
    root.attributes('-alpha', 0.8)
    root.title("Выберите фото")

    file_button = tk.Button(root, text="Выберите фото", command=open_file_dialog)
    file_button.pack(pady=20)
    file_label = tk.Label(root, text="")
    file_label.pack()
    root.mainloop()
    root.quit()

    return file_path

image_path = select_image_file()

image_tensor = image_to_tensor(image_path)

print("Размер тензора: ", image_tensor.shape)
print(image_tensor)

plt.imshow(image_tensor.numpy())
plt.axis("off")
plt.show()

input_image = load_and_prepare_image(image_path)
input_image = tf.expand_dims(input_image, axis=0)

model = tf.keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)

predictions = model.predict(input_image)

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

print(f"{Fore.BLUE}------------------------------------")
print(f"Топ 5 предсказаний:{Style.RESET_ALL}")

for i, (imagenet_id, label, score) in enumerate(decode_predictions):
    result_transleate = translator.translate(label, dest='ru')

    if 0.5 <= score <= 1.0:
        print(f"{i+1}: {Fore.GREEN}{label} - {result_transleate.text} ({score:.2f}){Style.RESET_ALL}")

    elif 0.3 <= score <= 0.5:
        print(f"{i+1}: {Fore.YELLOW}{label} - {result_transleate.text} ({score:.2f}){Style.RESET_ALL}")

    elif 0.0 <= score <= 0.3:
        print(f"{i+1}: {Fore.RED}{label} - {result_transleate.text} ({score:.2f}){Style.RESET_ALL}")

print(f"{Fore.BLUE}------------------------------------{Style.RESET_ALL}")
