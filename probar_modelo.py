import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- CONFIGURACIÓN ---
ruta_modelo = "SPtoIT"
print(f"Cargando modelo...")

config = PeftConfig.from_pretrained(ruta_modelo)
model_base = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    attn_implementation="eager" 
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model_base, ruta_modelo)

if torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"
model = model.to(device)
model.eval()

# --- FUNCIÓN DE TRADUCCIÓN ---
def obtener_atencion_capa_2(texto, src_lang="spa_Latn", tgt_lang="ita_Latn"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texto, return_tensors="pt").to(device)
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, forced_bos_token_id=forced_bos_id, max_length=50
        )
        traduccion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        tokenizer.tgt_lang = tgt_lang 
        target_inputs = tokenizer(text_target=traduccion, return_tensors="pt", add_special_tokens=True).to(device)
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=target_inputs["input_ids"],
            output_attentions=True
        )
        
        # OBTENEMOS SOLO LA CAPA 2 (Índice 2)
        # Promediamos las cabezas de atención (dim 0)
        atencion_capa_2 = outputs.cross_attentions[2][0].mean(dim=0).cpu().numpy()
        
        return traduccion, atencion_capa_2, inputs, target_inputs

# --- GRAFICAR CAPA ÚNICA ---
def graficar_capa_2(matrix, inputs, target_inputs):
    # 1. Obtener tokens
    src_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tgt_tokens = tokenizer.convert_ids_to_tokens(target_inputs["input_ids"][0])
    
    # Limpieza de texto (quitar guiones bajos de SentencePiece)
    src_tokens = [t.replace(' ', '') for t in src_tokens]
    tgt_tokens = [t.replace(' ', '') for t in tgt_tokens]
    
    # 2. RECORTAR BORDES (Quitar <s> y </s>)
    # Esto es vital para eliminar el ruido de la escala
    matrix_clean = matrix[1:-1, 1:-1]
    src_clean = src_tokens[1:-1]
    tgt_clean = tgt_tokens[1:-1]

    # 3. NORMALIZACIÓN POR FILA
    mins = matrix_clean.min(axis=1, keepdims=True)
    maxs = matrix_clean.max(axis=1, keepdims=True)
    norm_matrix = (matrix_clean - mins) / (maxs - mins + 1e-10)

    # 4. Graficar
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        norm_matrix, 
        xticklabels=src_clean, 
        yticklabels=tgt_clean, 
        cmap="Blues", 
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='lightgray'
    )
    plt.title("Alineación de Traducción (Capa 2)")
    plt.xlabel("Español")
    plt.ylabel("Italiano")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# --- EJECUCIÓN ---
frase = "Necesito hacer un modelo de traducción automática que funcione bien para varios idiomas"
print(f"Entrada: {frase}")

trad, matriz, inp, tgt = obtener_atencion_capa_2(frase)
print(f"Salida:  {trad}")

graficar_capa_2(matriz, inp, tgt)