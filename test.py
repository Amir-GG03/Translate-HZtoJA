import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. RUTA DONDE GUARDASTE EL MODELO
# Asegúrate de que la carpeta "SPtoIT" esté en el mismo lugar que este script
ruta_modelo = "SPtoIT"
config = PeftConfig.from_pretrained(ruta_modelo)

# 2. CARGAR EL MODELO BASE
# CAMBIO 1: Usamos torch.float32 para máxima estabilidad en Mac M2
model_base = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 3. FUSIONAR TU ENTRENAMIENTO (LoRA)
model = PeftModel.from_pretrained(model_base, ruta_modelo)

# CAMBIO 2: Detección correcta de hardware para Mac (MPS)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Usando: {device}")

model = model.to(device)
model.eval()

# --- FUNCIÓN DE TRADUCCIÓN ---
def traducir(texto, idioma_origen="spa_Latn", idioma_destino="ita_Latn"):
    tokenizer.src_lang = idioma_origen
    inputs = tokenizer(texto, return_tensors="pt").to(device)
    
    forced_bos_id = tokenizer.convert_tokens_to_ids(idioma_destino)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            max_length=100,
            num_beams=5,
            early_stopping=True
        )
    
    resultado = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return resultado

# --- PRUEBAS ---
frases = [
    "Soy un tonto, me voy a matar."
]

print("\nDE ESPAÑOL A ITALIANO\n")
for frase in frases:
    traduccion = traducir(frase)
    print(f"{frase}")
    print(f"{traduccion}")
    print("-" * 20)