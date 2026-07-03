import torch
import argparse
from datasets import load_dataset, Features, Dataset, Image, Value
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel

PUBLISH_TO_HUB = False

def parse_args():
    
    parser = argparse.ArgumentParser(description="Generate text from an image with nanoVLM")
    parser.add_argument("hf_model", type=str, help="HuggingFace repo ID to download from incase --checkpoint isnt set.")
    parser.add_argument("hf_dataset", type=str, help="Hf VQA dataset to test")
    parser.add_argument("batch_size", type=int, default=20, help="Eval batch size")
    parser.add_argument("hf_target_dataset", type=str, help="Hf VQA dataset to test")
    parser.add_argument("--prompt", type=str, default="Describe the art in this image", help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=1, help="Num. of outputs to generate")
    parser.add_argument("--eos_token_id", type=str, default=".", help="EOS token id to stop generation")
    return parser.parse_args()


def publish_to_hub(images, answers, ground_truths, hf_target_dataset):
    
    min_len = min(len(images), len(answers), len(ground_truths))
    
    images = images[:min_len]
    answers = answers[:min_len]
    ground_truths = ground_truths[:min_len]
    
    features = Features({
        "image": Image(),
        "answer": Value("string"),
        "ground_truth": Value("string"),
    })

    dataset = Dataset.from_dict(
        {
            "image": images,
            "answer": answers,
            "ground_truth": ground_truths,
        },
        features=features,
    )

    print(dataset)
    print(dataset.features)
    
    dataset.push_to_hub(
        hf_target_dataset,
        private=True,
    )
    
    

def first_sentence_filter(gen, eos_token="."):
    return [item.split(eos_token)[0] for item in gen]

def eos_remove_filter(gen, eos_token=".<|endoftext|>"):
    return [item.split(eos_token)[0] for item in gen]

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
    tokens: torch.Tensor = encoded["input_ids"].to(device)
    
    tokens = tokens.repeat(args.batch_size, 1)
    
    raw_dataset = load_dataset(args.hf_dataset, name="default", split="train")
    vqa_dataset = VQADataset(raw_dataset, tokenizer, image_processor)

    vqa_collator = VQACollator(tokenizer, model.cfg.lm_max_length)
    test_loader = DataLoader(
        vqa_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=vqa_collator,
        pin_memory=True,
    )
    
    images_pool = []
    generated_content = []
    ground_truth = []
    art_names = []

    for batch in tqdm(test_loader):

        images: list[torch.Tensor] = batch["image"].to(device)
        
        for img in images:
            detached_img = img.cpu().detach().numpy()
            detached_img = detached_img.transpose(1, 2, 0)
            detached_img = detached_img * 255
            detached_img = detached_img.astype("uint8")
            images_pool.append(detached_img)
        
        """ to print 
        
        image = images[0].cpu().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = image * 255
        image = image.astype("uint8")
        from PIL import Image
        Image.fromarray(image).show()
        """
        try:
            
            if tokens.shape[0] != images.shape[0]:
                tokens = tokens[:tokens.shape[0], ...]
            
            gen = model.generate(tokens, images, max_new_tokens=100, greedy=True)
            out = tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            out = first_sentence_filter(out)
            answer = eos_remove_filter(batch['answers'])
            
            generated_content.extend(out)
            ground_truth.extend(answer)
            
            if 'extra' in batch.keys():
                for i in range(len(images)):
                    art_names.append(batch['extra'][i]["art_name"])
            
        except Exception as e:
            print(e)
            continue
        
    print()
        
    if PUBLISH_TO_HUB:
        print("Publish to hub")
        publish_to_hub(images_pool, generated_content, ground_truth, args.hf_target_dataset)

if __name__ == "__main__":
    main()