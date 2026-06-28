import torch
import argparse
import datasets
from torch.utils.data import DataLoader
from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel

def parse_args():
    
    parser = argparse.ArgumentParser(description="Generate text from an image with nanoVLM")
    parser.add_argument("hf_model", type=str, help="HuggingFace repo ID to download from incase --checkpoint isnt set.")
    parser.add_argument("hf_dataset", type=str, help="Hf VQA dataset to test")
    parser.add_argument("--prompt", type=str, default="Describe the art in this image", help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=1, help="Num. of outputs to generate")
    parser.add_argument("--eos_token_id", type=str, default=".", help="EOS token id to stop generation")
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    model = VisionLanguageModel.from_pretrained(args.hf_model)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)
    
    template = f"Question: {args.prompt} Answer:"
    encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded["input_ids"].to(device)
    
    raw_dataset = datasets.load_dataset(args.hf_dataset, name="default", split="train")
    vqa_dataset = VQADataset(raw_dataset, tokenizer, image_processor)

    vqa_collator = VQACollator(tokenizer, model.cfg.lm_max_length)
    test_loader = DataLoader(
        vqa_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=vqa_collator,
        pin_memory=True,
    )

    for i, batch in enumerate(test_loader):

        images = batch["image"].to(device)

        print(template)
        for i in range(args.generations):
            gen = model.generate(tokens, images, max_new_tokens=50)
            out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            print(f"  >> Generation {i+1}: {out}")

if __name__ == "__main__":
    main()