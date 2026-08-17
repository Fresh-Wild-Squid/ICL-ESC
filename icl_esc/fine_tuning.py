"""LoRA and TRL fine-tuning orchestration for the ESConv project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "unsloth/Llama-3.1-8B-bnb-4bit"


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    random_state: int = 3407
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TrainingSettings:
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: float = 1
    max_steps: int = -1
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "ft-model"
    report_to: str = "none"
    packing: bool = False


@dataclass(frozen=True)
class FineTuningResult:
    """Artifacts returned by the end-to-end fine-tuning workflow."""

    model: Any
    tokenizer: Any
    trainer: Any
    trainer_stats: Any
    adapter_path: Path
    train_examples: int


def add_lora_adapters(model, settings: LoraSettings | None = None):
    """Attach Unsloth LoRA adapters to a base model."""
    from unsloth import FastLanguageModel

    settings = settings or LoraSettings()
    return FastLanguageModel.get_peft_model(
        model,
        r=settings.rank,
        target_modules=list(settings.target_modules),
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=settings.random_state,
        use_rslora=False,
        loftq_config=None,
    )


def build_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    settings: TrainingSettings | None = None,
):
    """Create a configured TRL SFT trainer without starting training."""
    from trl import SFTConfig, SFTTrainer

    settings = settings or TrainingSettings()
    config: dict[str, Any] = {
        "per_device_train_batch_size": settings.per_device_train_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "warmup_steps": settings.warmup_steps,
        "num_train_epochs": settings.num_train_epochs,
        "max_steps": settings.max_steps,
        "learning_rate": settings.learning_rate,
        "logging_steps": settings.logging_steps,
        "optim": settings.optim,
        "weight_decay": settings.weight_decay,
        "lr_scheduler_type": settings.lr_scheduler_type,
        "seed": settings.seed,
        "output_dir": settings.output_dir,
        "report_to": settings.report_to,
        # TRL 0.22.x keeps data-preparation options on SFTConfig.
        "dataset_text_field": "text",
        "max_length": settings.max_seq_length,
        "packing": settings.packing,
    }
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(**config),
    )


def save_adapters(model, tokenizer, output_path: str | Path) -> Path:
    """Save LoRA adapters and tokenizer to the same directory."""
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


def run_fine_tuning(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str | Path = "model",
    split: str = "train",
    context_window: int = 4,
    load_in_4bit: bool = True,
    lora_settings: LoraSettings | None = None,
    training_settings: TrainingSettings | None = None,
) -> FineTuningResult:
    """Run the notebook's complete ESConv LoRA fine-tuning workflow."""
    from unsloth import FastLanguageModel

    from .data_preparation import load_esconv_dataset, prepare_train_dataset
    from .prompts import configure_llama3_tokenizer

    training_settings = training_settings or TrainingSettings()
    output_path = Path(output_dir)
    training_settings = replace(training_settings, output_dir=str(output_path))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=training_settings.max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    configure_llama3_tokenizer(tokenizer)

    dataset = load_esconv_dataset()
    train_dataset = prepare_train_dataset(
        dataset,
        tokenizer,
        split=split,
        context_window=context_window,
    )
    print(
        f"Prepared {len(train_dataset):,} training examples "
        f"from {len(dataset[split]):,} ESConv dialogues."
    )

    model = add_lora_adapters(model, lora_settings)
    trainer = build_sft_trainer(
        model,
        tokenizer,
        train_dataset,
        training_settings,
    )
    trainer_stats = trainer.train()
    adapter_path = save_adapters(model, tokenizer, output_path)

    return FineTuningResult(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        trainer_stats=trainer_stats,
        adapter_path=adapter_path,
        train_examples=len(train_dataset),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 on ESConv with Unsloth LoRA and TRL 0.22.2."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--output-dir",
        default="model",
        help=(
            "Shared directory for Trainer checkpoints, final LoRA adapters, and "
            "the tokenizer (default: model)"
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> FineTuningResult:
    """CLI entry point; notebook fine-tuning cells remain available separately."""
    args = _build_argument_parser().parse_args(argv)
    training_settings = TrainingSettings(
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        packing=args.packing,
    )
    lora_settings = LoraSettings(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        random_state=args.seed,
    )
    result = run_fine_tuning(
        model_name=args.model_name,
        output_dir=args.output_dir,
        split=args.split,
        context_window=args.context_window,
        load_in_4bit=args.load_in_4bit,
        lora_settings=lora_settings,
        training_settings=training_settings,
    )
    print(f"Saved LoRA adapters and tokenizer to: {result.adapter_path.resolve()}")
    return result


if __name__ == "__main__":
    main()
