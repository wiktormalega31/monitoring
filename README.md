# Monitoring Twarzy

Aplikacja webowa do rozpoznawania twarzy w czasie rzeczywistym za pomocą Flask, OpenCV i Face Recognition.

## Wymagania

Przed uruchomieniem projektu upewnij się, że masz zainstalowane wymagane zależności. Możesz je zainstalować za pomocą:

```bash
pip install -r requirements.txt
```

## Funkcjonalności

- Rozpoznawanie twarzy w czasie rzeczywistym z kamery.
- Rejestrowanie nowej twarzy jako "Trusted".
- Wyświetlanie strumienia wideo w przeglądarce.

## Uruchamianie

1. Uruchom aplikację za pomocą:

```bash
python app.py
```

2. Otwórz przeglądarkę i przejdź pod adres `http://127.0.0.1:5000`.

## API

- **`GET /`**: Strona główna.
- **`GET /video`**: Strumień wideo z rozpoznawaniem twarzy.
- **`POST /register`**: Rejestracja nowej twarzy.

## Struktura projektu

```
monitoring/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── known_faces/
    └── trusted.jpg
```

## Autor
Wiktor Malęga

