import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image_bgr = cv.imread('Resonancia.jpg') 
image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

# Mostrar la imagen Cargada
plt.imshow(image)
plt.title('Imágen cargada')
plt.axis('off')
plt.show()

# Cargar la imagen en escala de grises
img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.title('Imagen escala de grises')
plt.axis('off')
plt.show()

# Dimensiones de la imagen
rows, cols = img_gray.shape

# ========================#
# EJERCICIO 1: TRASLACIÓN #
# ========================#

# Traslación entera (50, 30)
M1 = np.float32([[1, 0, 50], [0, 1, 30]])
translated_int = cv.warpAffine(img_gray, M1, (cols, rows))

# Traslación decimal (20.5, 15.5)
M2 = np.float32([[1, 0, 20.5], [0, 1, 15.5]])
translated_dec = cv.warpAffine(img_gray, M2, (cols, rows))

# Mostrar resultados de traslación
fig1, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img_gray, cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(translated_int, cmap='gray')
axs[1].set_title("Traslación (50, 30)")
axs[1].axis('off')

axs[2].imshow(translated_dec, cmap='gray')
axs[2].set_title("Traslación (20.5, 15.5)")
axs[2].axis('off')

plt.tight_layout()
plt.show()

# ======================#
# EJERCICIO 2: ROTACIÓN #
# ======================#

# Calcular el centro de la imagen
center = (cols / 2, rows / 2)

# Matriz de rotación (45 grados)
M_rot = cv.getRotationMatrix2D(center, 45, 1.0)
rotated = cv.warpAffine(img_gray, M_rot, (cols, rows))

# Mostrar imagen original y rotada
fig2, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(img_gray, cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(rotated, cmap='gray')
axs[1].set_title("Rotación 45°")
axs[1].axis('off')

plt.tight_layout()
plt.show()

# ====================#
# EJERCICIO 3: ESCALA #
# ====================#

# Escalado al 150%
scaled_up = cv.resize(img_gray, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)

# Escalado al 50%
scaled_down = cv.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

# Mostrar imágenes escaladas
fig3, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img_gray, cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')
print('La imágen original tiene un tamaño de:', img_gray.shape)

axs[1].imshow(scaled_up, cmap='gray')
axs[1].set_title("Escala 150%")
axs[1].axis('off')
print('La imágen aumentada tiene un tamaño de:', scaled_up.shape)

axs[2].imshow(scaled_down, cmap='gray')
axs[2].set_title("Escala 50%")
axs[2].axis('off')
print('La imágen reducida tiene un tamaño de:', scaled_down.shape)

plt.tight_layout()
plt.show()

