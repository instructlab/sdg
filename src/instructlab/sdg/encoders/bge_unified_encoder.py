# Standard
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Union
import os

# Third Party
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch


# Define a TypedDict for model configuration
class ModelConfig(TypedDict):
    pooling_method: str
    normalize_embeddings: bool
    max_length: int
    default_instruction: str
    batch_size: int


# Model-specific configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "BAAI/bge-base-en": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 512,
        "default_instruction": "Represent this sentence for searching relevant passages:",
        "batch_size": 256,
    },
    "BAAI/bge-base-en-v1.5": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 512,
        "default_instruction": "Represent this sentence for searching relevant passages:",
        "batch_size": 256,
    },
    "BAAI/bge-large-en": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 512,
        "default_instruction": "Represent this sentence for searching relevant passages:",
        "batch_size": 256,
    },
    "BAAI/bge-large-en-v1.5": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 512,
        "default_instruction": "Represent this sentence for searching relevant passages:",
        "batch_size": 256,
    },
    "BAAI/bge-m3": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 4096,
        "default_instruction": "Use the following sentences to search for relevant passages:",
        "batch_size": 32,
    },
    "BAAI/bge-multilingual-gemma2": {
        "pooling_method": "last_token",
        "normalize_embeddings": True,
        "max_length": 4096,
        "default_instruction": "Represent this for searching:",
        "batch_size": 20,
    },
}


@dataclass
class EncoderConfig:
    model_name: str
    model_config: ModelConfig
    device: torch.device
    num_gpus: int
    batch_size: int
    use_default_instruction: bool


class UnifiedBGEEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[torch.device] = None,
        use_fp16: bool = True,
        use_default_instruction: bool = True,
    ) -> None:
        """
        Unified encoder supporting all BGE model variants with model-specific configurations.
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model {model_name} not supported. Supported models: {list(MODEL_CONFIGS.keys())}"
            )

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        batch_size = MODEL_CONFIGS[model_name]["batch_size"]
        batch_size = batch_size * num_gpus if num_gpus > 0 else batch_size

        self.cfg = EncoderConfig(
            model_name=model_name,
            model_config=MODEL_CONFIGS[model_name],
            device=device,
            num_gpus=num_gpus,
            batch_size=batch_size,
            use_default_instruction=use_default_instruction,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(self.cfg.device)

        if self.cfg.num_gpus > 1:
            print(f"Using {self.cfg.num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)

    def _pool_embeddings(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Model-specific pooling method"""
        if self.cfg.model_config["pooling_method"] == "cls":
            return hidden_states[:, 0]
        if self.cfg.model_config["pooling_method"] == "mean":
            s = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        if self.cfg.model_config["pooling_method"] == "last_token":
            left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return hidden_states[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            return hidden_states[
                torch.arange(batch_size, device=hidden_states.device), sequence_lengths
            ]
        raise ValueError(
            f"Unknown pooling method: {self.cfg.model_config['pooling_method']}"
        )

    def _prepare_inputs(
        self,
        texts: Union[str, List[str]],
        instruction: str = "",
        query_description: str = "",
    ) -> List[str]:
        """Prepare inputs with model-specific formatting"""
        if isinstance(texts, str):
            texts = [texts]

        if (
            not instruction
            and self.cfg.use_default_instruction
            and self.cfg.model_config["default_instruction"]
        ):
            instruction = str(self.cfg.model_config["default_instruction"])

        if instruction:
            if "bge-multilingual" in self.cfg.model_name.lower():
                texts = [
                    f"<instruct>{instruction}\n{query_description}{text}"
                    for text in texts
                ]
            elif "bge-m3" not in self.cfg.model_name.lower():
                texts = [f"{instruction} {text}" for text in texts]
        return texts

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str]],
        instruction: str = "",
        query_description: str = "",
        show_progress: bool = True,
        return_tensors: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encode texts into embeddings."""
        input_was_string = isinstance(inputs, str)
        inputs = self._prepare_inputs(inputs, instruction, query_description)

        encodings = self.tokenizer(
            inputs,
            max_length=self.cfg.model_config["max_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(self.cfg.device)

        embeddings_list = []
        for i in tqdm(
            range(0, len(inputs), int(self.cfg.batch_size)),
            disable=not show_progress or len(inputs) < 256,
        ):
            batch = {k: v[i : i + self.cfg.batch_size] for k, v in encodings.items()}
            outputs = self.model(**batch)
            hidden_states = outputs.last_hidden_state
            embeddings = self._pool_embeddings(hidden_states, batch["attention_mask"])

            if self.cfg.model_config["normalize_embeddings"]:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            embeddings_list.append(embeddings.cpu())
            del outputs, hidden_states, embeddings, batch
            torch.cuda.empty_cache()

        embeddings = torch.cat(embeddings_list, dim=0)
        if input_was_string:
            embeddings = embeddings[0]

        return embeddings if return_tensors else embeddings.numpy()

    def encode_queries(
        self, queries: Union[str, List[str]], instruction: str = "", **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """Specialized method for encoding queries"""
        return self.encode(queries, instruction=instruction, **kwargs)

    def encode_corpus(
        self, corpus: Union[str, List[str]], instruction: str = "", **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """Specialized method for encoding corpus documents"""
        return self.encode(corpus, instruction=instruction, **kwargs)
