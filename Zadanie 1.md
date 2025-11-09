import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Wczytanie obrazu rastrowego zdalnego - POPRAWNY BEZPOŚREDNI URL
url_obrazu = "https://upload.wikimedia.org/wikipedia/commons/f/f6/Lemon_8FruitAndFlower_wb.jpg"

try:
    # Pobieranie obrazu zdalnego do pamięci
    req = urllib.request.urlopen(url_obrazu)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    
    # Dekodowanie tablicy numpy na obraz (domyślnie BGR w cv2)
    obraz_oryginalny = cv2.imdecode(arr, cv2.IMREAD_COLOR)

except Exception as e:
    print(f"Błąd podczas wczytywania obrazu: {e}")
    exit()

if obraz_oryginalny is None:
    print("Nie udało się wczytać obrazu. Sprawdź URL.")
    exit()

# 2. Wyświetlenie obrazu oryginalnego
# Konwersja BGR (cv2) na RGB (matplotlib)
obraz_rgb = cv2.cvtColor(obraz_oryginalny, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(obraz_rgb)
plt.title('1. Obraz Oryginalny')
plt.axis('off')

# 3. Zmiana rozdzielczości (zmniejszenie o 50%) i konwersja na grayscale
wysokosc, szerokosc = obraz_oryginalny.shape[:2]

# Zmniejszenie rozdzielczości: szerokość/2, wysokość/2
nowa_rozdzielczosc = (szerokosc // 2, wysokosc // 2)
obraz_zmniejszony = cv2.resize(obraz_oryginalny, nowa_rozdzielczosc, interpolation=cv2.INTER_LINEAR)

# Konwersja na grayscale
obraz_grayscale = cv2.cvtColor(obraz_zmniejszony, cv2.COLOR_BGR2GRAY)

# 4. Obrót obrazu o 90 stopni w prawo
obraz_obrocony = np.rot90(obraz_grayscale, k=-1) 

# 5. Wyświetlenie obrazu wynikowego
plt.subplot(1, 2, 2)
plt.imshow(obraz_obrocony, cmap='gray') 
plt.title('5. Obraz Wynikowy (Grayscale, 50% Rozdzielczości, Obrót 90°)')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Wyświetlanie macierzy obrazu / w postaci tablicy liczb
print("-" * 50)
print(f"Rozdzielczość obrazu wynikowego: {obraz_obrocony.shape[1]}x{obraz_obrocony.shape[0]}")
print("6. Fragment macierzy obrazu wynikowego (pierwsze 5 wierszy i 10 kolumn):")
print(obraz_obrocony[:5, :10])
print("-" * 50)
