import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_model():
    """
    Stable Diffusion modelini yükler ve GPU desteğiyle çalıştırır.
    """
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # GPU desteği
    return pipe

def generate_image(pipe, input_image_path, output_image_path, prompt="", strength=0.75, guidance_scale=7.5):
    """
    Verilen giriş resmi ve parametreler ile Stable Diffusion kullanarak yeni bir resim oluşturur.
    
    Args:
    - pipe (StableDiffusionImg2ImgPipeline): Model pipeline'ı.
    - input_image_path (str): Giriş resminin dosya yolu.
    - output_image_path (str): Oluşturulan resmin kaydedileceği dosya yolu.
    - prompt (str): Modelin rehber olarak kullanacağı metin.
    - strength (float): Giriş resminin ne kadar değiştirilmesi gerektiğini kontrol eder.
    - guidance_scale (float): Modelin ne kadar yaratıcı olacağını belirler.
    """
    
    input_image = Image.open(input_image_path).convert("RGB")
    
    # Modeli çalıştır ve sonuç görüntüsünü oluştur
    output_image = pipe(prompt=prompt, init_image=input_image, strength=strength, guidance_scale=guidance_scale).images[0]
    
    # Sonucu kaydet
    output_image.save(output_image_path)
    print(f"Output image saved at {output_image_path}")
    
    return output_image

if __name__ == "__main__":
    # Modeli yükleyin
    pipe = load_model()

    # Girdi ve çıktı yollarını belirleyin
    input_image_path = "person.jpg"  # Giriş resmi dosya yolu (VS Code ortamında mevcut olmalı)
    output_image_path = "output_image.jpg"  # Çıktı resmi dosya yolu

    # Parametreleri ayarlayarak görüntü üretimi
    output_image = generate_image(pipe, input_image_path, output_image_path, 
                                  prompt="a person standing in a beautiful landscape", 
                                  strength=0.7, 
                                  guidance_scale=8.5)

    # Sonuç görüntüsünü göster
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()
