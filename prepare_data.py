import random

# --- CONFIGURACIÓN ---
# Nombres exactos de tus archivos descargados (cámbialos si es necesario)
SOURCE_FILE = "es-it.txt/OpenSubtitles.es-it.es"
TARGET_FILE = "es-it.txt/OpenSubtitles.es-it.it"

# Archivos de salida (los que usarás para entrenar)
OUT_SRC = "train_clean.es"
OUT_TGT = "train_clean.it"

# Cantidad de frases BUENAS que quieres obtener
# Para tu M2 Pro, 500,000 es un excelente punto de partida. 
# 1 Millón es el límite recomendado para iterar rápido.
TARGET_COUNT = 500000 

def is_valid(src_line, tgt_line):
    """Reglas de calidad para decidir si una frase sirve."""
    l1 = len(src_line.split())
    l2 = len(tgt_line.split())

    # 1. Descartar frases muy cortas (menos de 3 palabras) o muy largas (más de 100)
    # Las de 1 o 2 palabras suelen ser nombres o exclamaciones que no aportan gramática.
    if l1 < 3 or l2 < 3: return False
    if l1 > 100 or l2 > 100: return False

    # 2. Descartar desajustes de longitud (Ratio Filter)
    # Si una frase es el triple de larga que la otra, probablemente sea un mal subtítulo.
    if l1 > l2 * 3 or l2 > l1 * 3: return False

    return True

print("Leyendo archivos y mezclando datos... (esto puede tardar un poco)")

# Leemos los archivos en memoria (OpenSubtitles es texto plano, 
# leer 5-10 millones de líneas es manejable en tus 16GB si no abres 50 pestañas de Chrome).
with open(SOURCE_FILE, "r", encoding="utf-8") as fs, open(TARGET_FILE, "r", encoding="utf-8") as ft:
    src_lines = fs.readlines()
    tgt_lines = ft.readlines()

# Verificación de seguridad
assert len(src_lines) == len(tgt_lines), "¡Error! Los archivos tienen diferente número de líneas."

print(f"Total de líneas encontradas: {len(src_lines)}")

# Empaquetamos para mezclar sin perder la alineación
combined = list(zip(src_lines, tgt_lines))

# MEZCLAR ALEATORIAMENTE (Shuffle)
# Esto asegura que tengas variedad de temas.
random.seed(42) # Para reproducibilidad
random.shuffle(combined)

print("Filtrando y guardando...")

count = 0
with open(OUT_SRC, "w", encoding="utf-8") as f_out_s, open(OUT_TGT, "w", encoding="utf-8") as f_out_t:
    for src, tgt in combined:
        # Limpiamos espacios extra y saltos de línea
        src_clean = src.strip()
        tgt_clean = tgt.strip()

        # Aplicamos filtros de calidad
        if is_valid(src_clean, tgt_clean):
            f_out_s.write(src_clean + "\n")
            f_out_t.write(tgt_clean + "\n")
            count += 1
        
        # Paramos si ya tenemos suficientes
        if count >= TARGET_COUNT:
            break

print(f"¡Listo! Se han guardado {count} pares de oraciones de alta calidad.")
print(f"Archivos generados: {OUT_SRC} y {OUT_TGT}")