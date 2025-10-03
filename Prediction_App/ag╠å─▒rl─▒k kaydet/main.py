from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.intel.openvino import OVQuantizer, OVConfig
from neural_compressor.config import PostTrainingQuantConfig

model_dir = "Urunum_bekledigim_kalitede_degil/hata_bert_finetuned_model"
save_directory = "Urunum_bekledigim_kalitede_degil/hata/openvino_model_quantized"

# Model ve tokenizer yükleniyor
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Kuantizasyon ayarları
ov_config = OVConfig()
quant_config = PostTrainingQuantConfig(approach="static")

# Görev tipi belirtiliyor!
quantizer = OVQuantizer.from_pretrained(model, task="text-classification")

# Kuantizasyon işlemi
quantizer.quantize(
    tokenizer=tokenizer,
    ov_config=ov_config,
    quantization_config=quant_config,
    calibration_dataset=None,  # Örnek dataset eklenebilir
    save_directory=save_directory,
    save_onnx_model=True
)

print("Model başarıyla kuantize edildi ve kaydedildi.")
