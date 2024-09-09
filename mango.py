import os
import cv2
import numpy as np

# Hàm tính tỷ lệ cường độ màu đỏ trên tổng cường độ của 3 kênh (R, G, B)
def calculate_red_ratio(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    total_intensity = red + green + blue
    total_intensity[total_intensity == 0] = 1
    red_ratio = red / total_intensity
    avg_red_ratio = np.mean(red_ratio)
    return avg_red_ratio

# Đường dẫn tới thư mục chứa ảnh xoài
image_folder = 'Mango_resize'

# Danh sách lưu trữ các tỷ lệ màu đỏ và nhãn tương ứng
mango_data = []

# Tính tỷ lệ đỏ cho tất cả các ảnh trong thư mục
for image_path in os.listdir(image_folder):
    if image_path.endswith(".jpg") or image_path.endswith(".png"):
        full_image_path = os.path.join(image_folder, image_path)
        red_ratio = calculate_red_ratio(full_image_path)
        mango_data.append((red_ratio, full_image_path))


# Gán nhãn dựa trên ngưỡng
threshold = 0.378
sweet_mangoes = [img for img in mango_data if img[0] > threshold]
sour_mangoes = [img for img in mango_data if img[0] <= threshold]

# Chọn các ảnh cụ thể cho tập huấn luyện
sweet_image_names = ['2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', '19.png', '20.png', '21.png', '22.png']
sour_image_names = ['11.png', '12.png', '13.png', '14.png', '15.png', '16.png', '17.png', '18.png']

train_sweet = [img for img in sweet_mangoes if os.path.basename(img[1]) in sweet_image_names]
train_sour = [img for img in sour_mangoes if os.path.basename(img[1]) in sour_image_names]

# Chọn ảnh test
test_sweet = next((img for img in sweet_mangoes if os.path.basename(img[1]) == '1.png'), None)
test_sour = next((img for img in sour_mangoes if os.path.basename(img[1]) == '10.png'), None)

# Kiểm tra nếu chọn được quả xoài
if test_sweet is None or test_sour is None:
    raise ValueError("Không tìm thấy ảnh cụ thể trong danh sách quả ngọt hoặc quả chua.")

# Tính tỷ lệ đỏ cho các ảnh trong tập huấn luyện
train_sweet_red_ratios = [(os.path.basename(img[1]), img[0]) for img in train_sweet]
train_sour_red_ratios = [(os.path.basename(img[1]), img[0]) for img in train_sour]

# Dự đoán dựa trên tỷ lệ màu đỏ của mẫu test
test_sweet_prediction = "Ngọt" if test_sweet[0] > threshold else "Chua"
test_sour_prediction = "Ngọt" if test_sour[0] > threshold else "Chua"

# In kết quả
print("\nTập quả ngọt (Test):")
print(f"{'Ảnh':<20} {'Tỷ lệ màu đỏ':<15} {'Loại quả':<10}")
print(f"{os.path.basename(test_sweet[1]) :<20} {test_sweet[0]:<15.3f} {test_sweet_prediction:<10}")

print("\nTập quả chua (Test):")
print(f"{'Ảnh':<20} {'Tỷ lệ màu đỏ':<15} {'Loại quả':<10}")
print(f"{os.path.basename(test_sour[1]) :<20} {test_sour[0]:<15.3f} {test_sour_prediction:<10}")

print("\nTập quả ngọt (Train):")
print(f"{'Ảnh':<20} {'Tỷ lệ màu đỏ':<15} {'Loại quả':<10}")
for img_name, red_ratio in train_sweet_red_ratios:
    print(f"{img_name:<20} {red_ratio:<15.3f} Ngọt")

print("\nTập quả chua (Train):")
print(f"{'Ảnh':<20} {'Tỷ lệ màu đỏ':<15} {'Loại quả':<10}")
for img_name, red_ratio in train_sour_red_ratios:
    print(f"{img_name:<20} {red_ratio:<15.3f} Chua")