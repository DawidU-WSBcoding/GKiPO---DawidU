import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Wczytanie obrazu rastrowego zdalnego
url_obrazu = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Lemon.jpg/640px-Lemon.jpg"


# Użycie Request z nagłówkiem User-Agent, dla poprawnego działania
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(url_obrazu, headers=headers)

with urllib.request.urlopen(req) as response:
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
    lemons_original = cv2.imdecode(arr, cv2.IMREAD_COLOR)


# 2. Wyświetlenie obrazu original
# Konwersja BGR (cv2) na RGB (matplotlib)
lemons_rgb = cv2.cvtColor(lemons_original, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(lemons_rgb)
plt.title('Lemons Original') # Zmiana tytułu
plt.axis('off')


# 3. Zmiana rozdzielczości (zmniejszenie o 50%) i konwersja na grayscale
wysokosc, szerokosc = lemons_original.shape[:2] 

# Zmniejszenie rozdzielczości
nowa_rozdzielczosc = (szerokosc // 2, wysokosc // 2)
lemons_zmniejszony = cv2.resize(lemons_original, nowa_rozdzielczosc, interpolation=cv2.INTER_LINEAR) # Zmiana

# Konwersja na grayscale
lemons_grayscale = cv2.cvtColor(lemons_zmniejszony, cv2.COLOR_BGR2GRAY)


# 4. Obrót obrazu o 90 stopni w prawo
lemons_obrocony = np.rot90(lemons_grayscale, k=-1) 


# 5. Wyświetlenie obrazu wynikowego
plt.subplot(1, 2, 2)
plt.imshow(lemons_obrocony, cmap='gray') 
plt.title('Lemons Output (Grayscale, 50% Rozdzielczości, Obrót 90°)')
plt.axis('off')

plt.tight_layout()
plt.show()


# 6. Wyświetlanie macierzy obrazu / w postaci tablicy liczb
print("-" * 50)
print(f"Rozdzielczość lemons output: {lemons_obrocony.shape[1]}x{lemons_obrocony.shape[0]}")
print("6. Fragment macierzy lemons output (pierwsze 5 wierszy i 10 kolumn):")
print(lemons_obrocony[:5, :10])
print("-" * 50)
