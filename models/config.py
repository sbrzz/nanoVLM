from dataclasses import dataclass


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 224
    vit_n_heads: int = 1
    vit_dropout: float = 0.0
    vit_n_blocks: int = 1
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip-base-patch16-224'

    lm_hidden_dim: int = 72
    lm_inter_dim: int = 192
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49152
    lm_n_heads: int = 1
    lm_n_kv_heads: int = 1
    lm_dropout: float = 0.0
    lm_n_blocks: int = 1
    lm_attn_scaling: float = 1.0
    IMAGE_TOKEN_LENGTH: int = 49
    TOTAL_SEQUENCE_LENGTH: int = 128
    lm_max_length: int = TOTAL_SEQUENCE_LENGTH - IMAGE_TOKEN_LENGTH  # Maximum length for the language model, derived from TOTAL_SEQUENCE_LENGTH and IMAGE_TOKEN_LENGTH
    lm_use_tokens: bool = False  # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True  # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
    lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'
    lm_eos_token_id: int = 0

    mp_pixel_shuffle_factor: int = 2

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints/nanoVLM_siglip-base-patch16-224_mp2_SmolLM2-135M_1xGPU_simon_CA'
    hf_repo_name: str = 'nanoVLM'


@dataclass
class TrainConfig:
    lr_mp: float = 2e-3
    lr_backbones: float = 1e-5
    data_cutoff_idx: int = None
    val_ratio: float = 0.025
    batch_size: int = 60
    gradient_accumulation_steps: int = 1
    mmstar_batch_size: int = 32
    max_grad_norm: float = None
    eval_in_epochs: bool = True
    eval_interval: int = 10
    epochs: int = 5
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False  # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    # train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    # train_dataset_path: str = 'thomasgauthier/small-cauldron'
    train_dataset_path: str = 'sbrzz/ca_genai'
    # train_dataset_name: tuple[str, ...] = ("ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa", "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam", "iconqa", "infographic_vqa", "intergps", "localized_narratives", "mapqa", "multihiertt", "ocrvqa", "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql", "robut_wtq", "scienceqa", "screen2words", "st_vqa", "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa", "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr", "websight")
    train_dataset_name: tuple[str, ...] = ("default")
    test_dataset_path: str = "Lin-Chen/MMStar"
    wandb_entity: str = "HuggingFace"  # Indicate the entity to log to in wandb
    log_wandb: bool = False
