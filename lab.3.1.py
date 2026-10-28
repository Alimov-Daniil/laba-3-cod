import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
)

# 1. Завантаження зображення
# Шлях до файлу (використовуємо префікс r для коректної обробки слешів Windows)
image_path = r"C:\Users\alimo\OneDrive\Робочий стіл\Pyton\lab3\stones_1.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Файл не знайдено за шляхом: {image_path}")

# 2. Перетворення у відтінки сірого (L-mode)
img_gray = Image.open(image_path).convert("L")
img_gray_arr = np.array(img_gray)

# 3. Порогове перетворення (бінаризація)
# Поріг підібрано емпірично (130-140 добре виділяє контури світлих каменів)
threshold = 135
im_bin = (img_gray_arr > threshold).astype(int)

# 4. Створення структурного елемента: РОМБ (Варіант 1)
# Генеруємо ромб за допомогою манхеттенської відстані
radius = 4  # розмір ромба (можна змінювати від 2 до 6)
y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
struct_diamond = (np.abs(x) + np.abs(y) <= radius).astype(int)

print("Матриця структурного елемента (Ромб):")
print(struct_diamond)

# Ерозія (звуження світлих ділянок / видалення дрібного шуму)
im_eroded = binary_erosion(im_bin, structure=struct_diamond)

# 5. Дилатація після ерозії (відновлення розмірів основних каменів)
im_dilated = binary_dilation(im_eroded, structure=struct_diamond)

# 6. Морфологічне розмикання та замикання
# Opening: ерозія -> дилатація (прибирає дрібні виступи та тонкі перемички)
im_opened = binary_opening(im_bin, structure=struct_diamond)
# Closing: дилатація -> ерозія (заповнює дірки та дрібні проміжки всередині каменів)
im_closed = binary_closing(im_opened, structure=struct_diamond)

# 7. Візуалізація результатів для звіту
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img_gray_arr, cmap="gray")
axes[0, 0].set_title("1. Відтінки сірого")
axes[0, 0].axis("off")

axes[0, 1].imshow(im_bin, cmap="gray")
axes[0, 1].set_title(f"2. Бінарне (поріг = {threshold})")
axes[0, 1].axis("off")

axes[0, 2].imshow(struct_diamond, cmap="gray")
axes[0, 2].set_title("3. Структурний елемент (Ромб)")
axes[0, 2].axis("off")

axes[1, 0].imshow(im_eroded, cmap="gray")
axes[1, 0].set_title("4. Ерозія (Erosion)")
axes[1, 0].axis("off")

axes[1, 1].imshow(im_dilated, cmap="gray")
axes[1, 1].set_title("5. Дилатація після ерозії")
axes[1, 1].axis("off")

axes[1, 2].imshow(im_closed, cmap="gray")
axes[1, 2].set_title("6. Розмикання + Замикання")
axes[1, 2].axis("off")

plt.tight_layout()

# Збереження готового зведеного малюнка для звіту
output_dir = os.path.dirname(image_path)
output_file = os.path.join(output_dir, "lab3_results.png")
plt.savefig(output_file, dpi=300)
print(f"Результати збережено у: {output_file}")

plt.show()