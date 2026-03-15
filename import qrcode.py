import qrcode

# Dein Link
url = "https://data-science-projekt-2026.streamlit.app"

# QR-Code erstellen
qr = qrcode.QRCode(box_size=10, border=4)
qr.add_data(url)
qr.make(fit=True)

# Als Bild speichern
img = qr.make_image(fill_color="black", back_color="white")
img.save("figures/qrcode.png")