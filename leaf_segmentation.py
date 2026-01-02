import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Tukar nama file di sini
img_path = "leaf.jpeg"

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Tak jumpa leaf.jpeg. Pastikan file ada dalam folder yang sama.")

# Display original
plt.figure(figsize=(6,10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Figure 1: Original Image")
plt.axis("off")
plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([25, 40, 40])
upper_green = np.array([95, 255, 255])

mask_raw = cv2.inRange(hsv, lower_green, upper_green)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise RuntimeError("Contour tak jumpa. Adjust HSV range atau pastikan background plain.")

leaf_cnt = max(contours, key=cv2.contourArea)
leaf_area_px = cv2.contourArea(leaf_cnt)
print("Leaf Area (pixels):", int(leaf_area_px))

result = img.copy()
x, y, w, h = cv2.boundingRect(leaf_cnt)
cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.drawContours(result, [leaf_cnt], -1, (0, 255, 0), 2)
cv2.putText(result, f"Leaf Area: {int(leaf_area_px)} px", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

segmented = cv2.bitwise_and(img, img, mask=mask_clean)

# Save outputs
os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/01_original.jpg", img)
cv2.imwrite("outputs/02_mask_raw.jpg", mask_raw)
cv2.imwrite("outputs/03_mask_clean.jpg", mask_clean)
cv2.imwrite("outputs/04_result_annotated.jpg", result)
cv2.imwrite("outputs/05_segmented_leaf.jpg", segmented)

print("Done. Output saved in outputs/")