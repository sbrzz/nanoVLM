import pathlib

import numpy as np
import torch
import argparse
from pathlib import Path
from datetime import datetime
from datasets import load_dataset, Features, Dataset, Image, Value
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_inference_augmenter, get_tokenizer
from models.vision_language_model import VisionLanguageModel

PUBLISH_TO_HUB = False
ENABLE_GREEDY = False
ENABLE_STOCHASTIC = True
ENABLE_AUGMENTED = False

def parse_args():
    
    parser = argparse.ArgumentParser(description="Generate text from an image with nanoVLM")
    parser.add_argument("hf_model", type=str, help="HuggingFace repo ID to download from incase --checkpoint isnt set.")
    parser.add_argument("hf_dataset", type=str, help="Hf VQA dataset to test")
    parser.add_argument("batch_size", type=int, default=20, help="Eval batch size")
    parser.add_argument("hf_target_dataset", type=str, help="Hf VQA dataset to test")
    parser.add_argument("tmp_dir", type=str, help="Path to tmp dir")
    parser.add_argument("--prompt", type=str, default="Describe the art in this image", help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=1, help="Num. of outputs to generate")
    parser.add_argument("--eos_token_id", type=str, default=".", help="EOS token id to stop generation")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=0, help="Top k for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p for sampling")
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
    augmenter = get_inference_augmenter(model.cfg.vit_img_size)
    
    template = f"Question: {args.prompt} Answer:"
    encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens: torch.Tensor = encoded["input_ids"].to(device)
    
    tokens = tokens.repeat(args.batch_size, 1)
    
    raw_dataset = load_dataset(args.hf_dataset, name="default", split="train")
    vqa_dataset = VQADataset(raw_dataset, tokenizer, image_processor, augmenter)

    vqa_collator = VQACollator(tokenizer, model.cfg.lm_max_length)
    test_loader = DataLoader(
        vqa_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=vqa_collator,
        pin_memory=True,
    )
    
    images_pool = []
    generated_content_greedy = []
    generated_content_stochastic = []
    generated_content_augmented = []
    ground_truth = []
    art_names = []

    for batch in tqdm(test_loader):

        images: list[torch.Tensor] = batch["image"].to(device)
        augmented_images: list[torch.Tensor] = batch["augmented_image"].to(device)
        
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
                
                
            if ENABLE_GREEDY:
            
                """ Greedy generation """
                
                gen, mp_embedding = model.generate(tokens, images, max_new_tokens=100, greedy=True, return_mp_embedding=True)
                out_greedy = tokenizer.batch_decode(gen, skip_special_tokens=True)
                
                out_greedy = first_sentence_filter(out_greedy)
                generated_content_greedy.extend(out_greedy)
                
                # save mp_embedding
                mp_embedding = mp_embedding.cpu().detach().numpy()
                
                if 'extra' in batch.keys():
                    for i in range(len(images)):
                        current_name = batch['extra'][i]["art_name"]
                        current_name = current_name.replace(" ", "_")
                        current_embedding = mp_embedding[i, :]
                        stem = datetime.now().strftime('%Y%m%d%H%M%S%f')
                        dest_path = pathlib.Path(args.tmp_dir) / "mp_embedding" / current_name / f"{stem}.npy"
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(dest_path, current_embedding.reshape(-1))
                        
            if ENABLE_STOCHASTIC:
            
                """ Stochastic generation """
                
                params = {
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                }
                
                gen = model.generate(tokens, images, max_new_tokens=100, greedy=False, **params)
                out_stochastic = tokenizer.batch_decode(gen, skip_special_tokens=True)
                
                out_stochastic = first_sentence_filter(out_stochastic)
                generated_content_stochastic.extend(out_stochastic)
            
            if ENABLE_AUGMENTED:
            
                """ Augmented generation """
                
                gen = model.generate(tokens, augmented_images, max_new_tokens=100)
                out_augmented = tokenizer.batch_decode(gen, skip_special_tokens=True)
                
                out_augmented = first_sentence_filter(out_augmented)
                generated_content_augmented.extend(out_augmented)
            
            """ Other stuff """
                
            answer = eos_remove_filter(batch['answers'])
            ground_truth.extend(answer)
            
            if 'extra' in batch.keys():
                for i in range(len(images)):
                    art_names.append(batch['extra'][i]["art_name"])
            
        except Exception as e:
            print(e)
            continue
        
    
    if ENABLE_GREEDY and ENABLE_STOCHASTIC and ENABLE_AUGMENTED:
        name_in_description_greedy_accuracy = 0
        name_in_description_stochastic_accuracy = 0
        name_in_description_gen_aug_accuracy = 0
        for gen_greedy, gen_stochastic, gen_aug, art_name in zip(generated_content_greedy, generated_content_stochastic, generated_content_augmented, art_names):
            if art_name in gen_greedy:
                name_in_description_greedy_accuracy += 1
            if art_name in gen_stochastic:
                name_in_description_stochastic_accuracy += 1
            if art_name in gen_aug:
                name_in_description_gen_aug_accuracy += 1
                
        print(f"Name in description greedy accuracy: {name_in_description_greedy_accuracy / len(generated_content_greedy)}")
        print(f"Name in description stochastic accuracy: {name_in_description_stochastic_accuracy / len(generated_content_greedy)}")
        print(f"Name in description gen_aug accuracy: {name_in_description_gen_aug_accuracy / len(generated_content_greedy)}")
    
    if ENABLE_STOCHASTIC:
        stem = datetime.now().strftime('%Y%m%d%H%M%S')
        with open(Path(args.tmp_dir) / f"{stem}_generated_content_stochastic.txt", "w") as f:
            
            f.write(str(params) + "\n")
            
            for item in generated_content_stochastic:
             
                item = item.encode('utf-8', 'ignore').decode('utf-8')
                
                try:
                    f.write(item + "\n")
                except Exception as e:
                    print(item)
                    print(e)
                    continue
        
    if PUBLISH_TO_HUB:
        print("Publish to hub")
        publish_to_hub(images_pool, gen_greedy, ground_truth, args.hf_target_dataset)

if __name__ == "__main__":
    main()