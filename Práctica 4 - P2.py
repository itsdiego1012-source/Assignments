import cv2 as cv
import matplotlib.pyplot as plt

# Cargar la imagen
image_bgr = cv.imread('Radiografia.jpg') 
image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

# Mostrar la imágen Cargada
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

# Función para graficar histograma
hist = cv.calcHist([img_gray], [0], None, [256], [0, 256])
plt.hist(img_gray.ravel(), 256, color='black')
plt.title("Histograma de Imagen en Escala de Grises")
plt.xlabel("Intensidad de Píxel")
plt.ylabel("Frecuencia")
plt.show()

#Función para Ecualizar el histograma
img_eq = cv.equalizeHist(img_gray)

# Mostrar imagen ecualizada
plt.imshow(img_eq, cmap='gray')
plt.title("Imagen ecualizada")
plt.axis("off")
plt.show()

# Graficar histograma de la imagen ecualizada
hist_eq = cv.calcHist([img_eq], [0], None, [256], [0, 256])
plt.hist(img_eq.ravel(), 256, color='black')
plt.title("Histograma de Imagen Ecualizada")
plt.xlabel("Intensidad de Píxel")
plt.ylabel("Frecuencia")
plt.show()

# Comparación de imágenes e histogramas
imagenes = [img_gray, img_eq]
titulos = ["Original", "Ecualizada"]
plt.figure(figsize=(16,8))

for i, (img, titulo) in enumerate(zip(imagenes, titulos)):
    # Mostrar imagenes
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(titulo)
    plt.axis("off")

    # Mostrar histograma
    plt.subplot(2, 2, i + 3)
    plt.hist(img.ravel(), 256, [0, 256], color='black')
    plt.title("Histograma " + titulo)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()
