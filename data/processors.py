from transformers import AutoTokenizer
import torchvision.transforms as transforms

TOKENIZERS_CACHE = {}

def get_tokenizer(name):
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

def get_image_processor(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

def get_inference_augmenter(img_size):  
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=2, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])