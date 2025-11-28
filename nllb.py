import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. CONFIGURACIÓN ---
MODEL_ID = "facebook/nllb-200-distilled-600M"
SRC_FILE = "train_clean.es"
TGT_FILE = "train_clean.it"
SRC_LANG = "spa_Latn" 
TGT_LANG = "ita_Latn"

# Configuración de Hardware (Mac M2)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# --- 2. PREPARACIÓN DE DATOS ---
print("📂 Cargando datos...")
dataset_src = load_dataset("text", data_files={"train": SRC_FILE})
dataset_tgt = load_dataset("text", data_files={"train": TGT_FILE})

full_dataset = dataset_src["train"].add_column("target_text", dataset_tgt["train"]["text"])
full_dataset = full_dataset.rename_column("text", "source_text")
dataset_splits = full_dataset.train_test_split(test_size=0.01)

# --- 3. TOKENIZACIÓN ---
print("⚙️ Preparando Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG)

def preprocess_function(examples):
    inputs = examples["source_text"]
    targets = examples["target_text"]
    tokenizer.src_lang = SRC_LANG
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    tokenizer.tgt_lang = TGT_LANG 
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset_splits.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset_splits["train"].column_names
)

# --- 4. MODELO + LoRA ---
print("🧠 Cargando Modelo NLLB y aplicando LoRA...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# IMPORTANTE: Desactivamos el cache para entrenamiento, pero 
# YA NO forzamos gradient_checkpointing_enable() para evitar tu error.
model.config.use_cache = False 

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=32,            
    lora_alpha=32,   
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 

# --- 5. ENTRENAMIENTO ---
print("Iniciando Entrenamiento (Modo Estabilizado)...")

training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-traductor-es-it",
    
    # --- CAMBIOS DE ESTABILIDAD ---
    learning_rate=5e-5,              # Bajamos la velocidad (antes 2e-4)
    warmup_steps=200,                # Calentamiento suave los primeros 200 pasos
    max_grad_norm=1.0,               # ¡CRÍTICO! Corta los picos de error para que no explote
    # ------------------------------

    # Configuración de Memoria (Misma de antes)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=False,    
    dataloader_num_workers=0,
    dataloader_pin_memory=False,     

    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=10,                # Bajamos a 10 para vigilarlo de cerca al inicio
    fp16=False,
    use_mps_device=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# --- 6. GUARDAR ---
print("💾 Guardando adaptador...")
model.save_pretrained("./modelo_final_lora")
tokenizer.save_pretrained("./modelo_final_lora")
print("¡Entrenamiento finalizado exitosamente!")