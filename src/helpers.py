from dataclasses import asdict, fields, is_dataclass

from model import GPTConfig


def training_to_gpt_config(train_cfg, vocab_size: int = 50304) -> GPTConfig:
    if is_dataclass(train_cfg):
        train_dict = asdict(train_cfg)
    else:
        train_dict = train_cfg  # already a dict

    train_dict = {k.lower(): v for k, v in train_dict.items()}
    gpt_fields = {f.name for f in fields(GPTConfig)}

    kwargs = {k: v for k, v in train_dict.items() if k in gpt_fields}
    kwargs["vocab_size"] = vocab_size

    return kwargs
