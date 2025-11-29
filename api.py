import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. CONFIGURACIÓN INICIAL (Se ejecuta al arrancar) ---
print("INIT SERVER")

# Ruta al modelo (carpeta local)
ruta_modelo = "SPtoIT"

# Detectar dispositivo (Optimizado para M2)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Metal?: {device}")

# Cargar configuración y tokenizador
config = PeftConfig.from_pretrained(ruta_modelo)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Cargar Modelo Base
model_base = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float32,  # float32 es lo mejor para MPS (Mac)
    low_cpu_mem_usage=True
)

# Cargar LoRA (Tu entrenamiento)
model = PeftModel.from_pretrained(model_base, ruta_modelo)
model = model.to(device)
model.eval()

# --- 2. DEFINICIÓN DE LA API ---
app = FastAPI(title="API Traductor NLLB-LoRA")

# Configurar CORS (Para que puedas llamarlo desde un frontend web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambia esto por tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de datos para validar lo que te envían
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "spa_Latn"  # Valores por defecto
    target_lang: str = "ita_Latn"

# --- 3. ENDPOINT (La función que recibe los datos) ---
@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Configurar idioma origen
        tokenizer.src_lang = request.source_lang
        
        # Procesar entrada y mover a GPU/MPS
        inputs = tokenizer(request.text, return_tensors="pt").to(device)
        
        # Obtener ID del idioma destino
        forced_bos_id = tokenizer.convert_tokens_to_ids(request.target_lang)
        
        # Generar traducción (Sin calcular gradientes)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=100,
                num_beams=5,
                early_stopping=True
            )
        
        # Decodificar
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return {
            "original": request.text,
            "translation": translation,
            "device": device
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))