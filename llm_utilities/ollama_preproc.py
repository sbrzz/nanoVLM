import os
import pathlib
import pickle
import datasets
import asyncio
import re
from loguru import logger
from tqdm import tqdm
from datasets import Dataset, Features, Value, Image as HFImage, Sequence, DatasetDict, concatenate_datasets
import ollama  # Ensure this supports async – if not, see notes below

NUMBER_OF_ANSWERS = 2

def upload_to_hf(__dataset, __dataset_name, __workdir):
    features = Features({
        "images": Sequence(feature=HFImage()),  # can be PIL.Image or path
        "texts": [{
            "user": Value("string"),
            "assistant": Value("string"),
            "source": Value("string"),
        }]
    })

    ds_info = datasets.DatasetInfo(description=__dataset_name, version="0.0.1", features=features)
    ds = Dataset.from_list(__dataset, info=ds_info, features=features)

    dict_ds = datasets.DatasetDict({"train": ds})

    try:
        logger.info("push to hub...")
        dict_ds.push_to_hub(f"cultural-arts/{__dataset_name}", private=True, token=os.getenv("HF_TOKEN"))
    except Exception as e:
        logger.error(f"Error: {e}")

    dict_ds.save_to_disk(str(__workdir / f"{__dataset_name}"))

    logger.info(f"Created dataset with {len(__dataset)} arts...")


def load_cache(__src_dir: pathlib.Path):

    __cached_files = list(__src_dir.glob("augmented*"))

    if len(__cached_files) == 0:
        return []
    
    __final_dataset = []
    for __cached_file in __cached_files:
        with open(__cached_file, "rb") as __f:
            __final_dataset.extend(pickle.load(__f))

    return __final_dataset


def save_dict(__ds, __target_hf):
    logger.info(f"Saving {len(__ds)} items to {__target_hf}...")
    with open(__target_hf, "wb") as f:
        pickle.dump(__ds, f)


def parse_response(response: str) -> list[str]:
    """Extract NUMBER_OF_ANSWERS numbered responses in one pass."""
    items = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+[\.\):]?\s*(.*)", line)
        if match and match.group(1):
            items.append(match.group(1).strip())
    return items


async def process_item(global_idx, item, model, prompt_name, prompt_intro):
    if item["art_name"]:
        prompt= prompt_name.format(seed_text=item["art_name"], NUMBER_OF_ANSWERS=NUMBER_OF_ANSWERS)
    else:
        prompt = prompt_intro.format(seed_text=item["intro_en"], NUMBER_OF_ANSWERS=NUMBER_OF_ANSWERS)

    try:
        response = await asyncio.to_thread(
            ollama.generate,
            model=model,
            prompt=prompt,
        )
        proposals = parse_response(response["response"])

        new_items = []
        for p in proposals:
            new_item = {
                "images": [item["image"]],
                "texts": [{
                    "user": "Describe the art in this image",
                    "assistant": p,
                    "source": "",
                }]
            }
            new_items.append(new_item)

        return new_items
    except Exception as e:
        logger.error(f"[{global_idx}] Error: {e}")
        return []
    

def save_batches(
    final_dataset: list,
    output_dir: pathlib.Path,
    batch_size: int = 1000,
):
    datasets_parts = []

    for start in range(0, len(final_dataset), batch_size):
        end = min(start + batch_size, len(final_dataset))

        batch = final_dataset[start:end]

        ds = Dataset.from_list(batch)

        shard_dir = output_dir / f"shard_{start:06d}_{end:06d}"
        ds.save_to_disk(str(shard_dir))

        datasets_parts.append(ds)

        logger.info(
            f"Saved shard {len(datasets_parts)} "
            f"({start}:{end}) with {len(batch)} samples"
        )

    merged = concatenate_datasets(datasets_parts)

    return DatasetDict({"train": merged})


async def main():

    working_dir = pathlib.Path(os.getenv("WORKINGDIR"))

    hf_dataset_dir = pathlib.Path(working_dir / "ca_raw_extended.pkl")
    template_target_hf = working_dir /  "augmented_hf_dataset_{idx}.pkl"

    if not hf_dataset_dir.exists():
        raise FileNotFoundError("Original dataset not found")
    
    final_dataset = load_cache(working_dir)

    dict_ds = save_batches(
        final_dataset,
        working_dir / "dataset_shards",
        batch_size=1000,
    )

    dict_ds.push_to_hub(
        "cultural-arts/ca_augmented_qwen25-14b-instruct",
        private=True,
        token=os.getenv("HF_TOKEN"),
    )

    if len(final_dataset) == 0:

        with open(hf_dataset_dir, "rb") as f:
            data_dicts = pickle.load(f)
            logger.info(f"Loaded {len(data_dicts)} items from pickle.")

        filtered_data_dicts = []
        for item in data_dicts:
            if item["art_name"] is None and item["intro_en"] is None:
                continue
            filtered_data_dicts.append(item)

        model = "qwen2.5:14b-instruct"
        prompt_intro = (
            'You are generating training answers for a cultural heritage app. The question being answered '
            'is always: "Describe the art in this image." Rewrite the following description into {NUMBER_OF_ANSWERS} distinct '
            'third-person answers to that question, written in the register of an art or heritage description. '
            'Preserve every factual detail exactly (names, dates, places, materials, '
            'events) — do not add, omit, or alter any fact. Vary sentence structure, word order, and phrasing '
            'across the {NUMBER_OF_ANSWERS} versions so no two read alike; vary length naturally (some short and direct, some '
            'longer), and vary which detail each sentence opens with. Output only a numbered list from 1 to {NUMBER_OF_ANSWERS}, '
            'with no preamble, no explanation, and no extra labels.\n\n'
            'Description: "{seed_text}"'
        )

        prompt_name = (
            'You are generating training answers for a cultural heritage app. The question being answered '
            'is always: "Describe the art in this image." Generate a numbered list of {NUMBER_OF_ANSWERS} distinct third-person '
            'answers to that question that identify the cultural artifact with name "{seed_text}", in the register of an '
            'art or heritage description (e.g. "This monument is known as...", "The place depicted here is '
            'titled...", "This is identified as..."), without stating or inventing any specific facts about it. '
            'Vary the sentence structure and phrasing naturally across the {NUMBER_OF_ANSWERS} versions, but keep every sentence '
            'factually neutral beyond the name itself. Output only a numbered list from 1 to {NUMBER_OF_ANSWERS}, with no '
            'preamble, no explanation, and no extra labels.'
        )

        buffer = []
        save_every = 100

        list_items: int = len(filtered_data_dicts)

        for idx in tqdm(range(list_items)):

            buffer.append(process_item(idx, filtered_data_dicts[idx], model, prompt_name, prompt_intro))

            # If buffer full or last item
            if len(buffer) >= save_every or idx == len(filtered_data_dicts) - 1:
                current_dataset = []
                logger.info(f"Saving {idx+1} items to {working_dir}")
                results = await asyncio.gather(*buffer)
                for generated_items in results:
                    current_dataset.extend(generated_items)
                save_path = pathlib.Path(str(template_target_hf).format(idx=idx))
                save_dict(current_dataset, save_path)

                final_dataset.extend(current_dataset)

                buffer.clear()

    # upload_to_hf(final_dataset, "ca_augmented_qwen25-14b-instruct", working_dir)


if __name__ == "__main__":
    asyncio.run(main())
