import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

print("INIT SERVER...")

# Configuración
ruta_modelo = "SPtoIT"

# Detección de Hardware
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

# Cargar Modelo Base (Con 'eager' para permitir extracción de atención)
model_base = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)

# Cargar LoRA 
model = PeftModel.from_pretrained(model_base, ruta_modelo)
model = model.to(device)
model.eval()

app = FastAPI(title="API Traductor NLLB-LoRA")

# Configuración CORS
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
        # 1. Configurar idiomas
        tokenizer.src_lang = request.source_lang
        tokenizer.tgt_lang = request.target_lang
        
        # 2. Procesar entrada
        inputs = tokenizer(request.text, return_tensors="pt").to(device)
        forced_bos_id = tokenizer.convert_tokens_to_ids(request.target_lang)
        
        with torch.no_grad():
            # A. Generar traducción
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=100,
                num_beams=5,
                early_stopping=True
            )
            
            # Decodificar texto
            translation_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            # B. Extraer Atención Cruzada (Cross-Attention) de la Capa 2
            # Tokenizamos el resultado para pasarlo como target
            target_inputs = tokenizer(
                text_target=translation_text, 
                return_tensors="pt",
                add_special_tokens=True
            ).to(device)

            # Forward pass manual
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=target_inputs["input_ids"],
                output_attentions=True 
            )
            
            # SELECCIÓN CRÍTICA: Capa 2 (Index 2)
            # Esta capa mostró la mejor alineación semántica en las pruebas
            layer_2_attention = outputs.cross_attentions[2][0]
            
            # Promediar las cabezas (heads)
            avg_attention = layer_2_attention.mean(dim=0)
            
            # Convertir a lista para JSON
            attention_matrix = avg_attention.cpu().numpy().tolist()

            # C. Preparar etiquetas (Tokens)
            src_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            src_tokens = [t.replace(' ', '') for t in src_tokens]
            
            tgt_tokens = tokenizer.convert_ids_to_tokens(target_inputs["input_ids"][0])
            tgt_tokens = [t.replace(' ', '') for t in tgt_tokens]

        return {
            "original": request.text,
            "translation": translation_text,
            "device": device,
            "attention": {
                "matrix": attention_matrix,
                "src_tokens": src_tokens,
                "tgt_tokens": tgt_tokens
            }
        }

    except Exception as e:
        print(f"Error detallado: {e}") 
        raise HTTPException(status_code=500, detail=str(e))