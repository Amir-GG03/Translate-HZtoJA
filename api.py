import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

print("INIT SERVER")

ruta_modelo = "SPtoIT"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Hardware: {device}")

# Cargar configuración y tokenizador
config = PeftConfig.from_pretrained(ruta_modelo)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Cargar Modelo Base
model_base = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    dtype=torch.float32, 
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)

# Cargar LoRA 
model = PeftModel.from_pretrained(model_base, ruta_modelo)
model = model.to(device)
model.eval()

app = FastAPI(title="API Traductor NLLB-LoRA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "spa_Latn" 
    target_lang: str = "ita_Latn"

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Configurar idioma origen
        tokenizer.src_lang = request.source_lang
        
        # Procesar entrada
        inputs = tokenizer(request.text, return_tensors="pt").to(device)
        
        # Obtener ID del idioma destino
        forced_bos_id = tokenizer.convert_tokens_to_ids(request.target_lang)
        
        with torch.no_grad():
            # 1. GENERAR LA TRADUCCIÓN
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=100,
                num_beams=5,
                early_stopping=True
            )
            
            # 2. ### NUEVO: GENERAR MATRIZ DE ATENCIÓN ###
            # Hacemos una pasada solo por el Encoder para ver las relaciones internas
            encoder_outputs = model.model.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
                return_dict=True
            )
            
            # encoder_outputs.attentions es una tupla (una por capa)
            # Tomamos la ÚLTIMA capa [-1], que tiene la información más semántica
            # Shape original: (batch=1, num_heads=16, seq_len, seq_len)
            last_layer_attention = encoder_outputs.attentions[-1][0] 
            
            # Promediamos las 16 cabezas para tener una sola matriz fácil de visualizar 2D
            # Shape resultante: (seq_len, seq_len)
            avg_attention = last_layer_attention.mean(dim=0)
            
            # Convertimos a lista de Python para que FastAPI lo pueda mandar como JSON
            attention_matrix = avg_attention.cpu().numpy().tolist()
            
            # Obtenemos los tokens (palabras) para que el front sepa qué etiquetar
            input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            # Limpiamos el caracter especial de SentencePiece " "
            input_tokens = [t.replace(' ', '') for t in input_tokens]

        # Decodificar
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return {
            "original": request.text,
            "translation": translation,
            "device": device,
            # ### NUEVO: Enviamos los datos extra al front
            "attention": {
                "matrix": attention_matrix, # Matriz 2D [[0.1, ...], [0.2, ...]]
                "tokens": input_tokens      # Etiquetas ["El", "perro", ...]
            }
        }

    except Exception as e:
        print(f"Error: {e}") # Imprimir error en consola servidor para debug
        raise HTTPException(status_code=500, detail=str(e))