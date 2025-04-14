# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Union
import logging
import os

# Third Party
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def safe_print(rank, msg):
    """Only print from rank 0."""
    if rank == 0:
        logger.info(msg)


# Define model configuration
class ModelConfig(TypedDict):
    pooling_method: str
    normalize_embeddings: bool
    max_length: int
    default_instruction: str
    batch_size: int


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 4096,
        "default_instruction": "Retrieve relevant passages:",
        "batch_size": 24,
    }
}


# pylint: disable=too-many-instance-attributes
@dataclass
class EncoderConfig:
    model_name: str
    model_config: ModelConfig
    device: torch.device
    num_gpus: int
    batch_size: int
    use_default_instruction: bool
    use_fp16: bool
    testing_mode: bool = False


class ArcticEmbedEncoder:
    def __init__(
        self,
        model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
        device: Optional[torch.device] = None,
        use_fp16: bool = False,
        use_default_instruction: bool = True,
        testing_mode: bool = False,
    ) -> None:
        """Initialize the Arctic encoder."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model {model_name} not supported. Supported models: {list(MODEL_CONFIGS.keys())}"
            )

        # Use the provided device or default to CUDA
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Get device ID for logging
        self.device_id = self.device.index if hasattr(self.device, "index") else 0

        # We don't need multi-GPU inside this encoder instance since each instance
        # will run on a dedicated GPU
        self.cfg = EncoderConfig(
            model_name=model_name,
            model_config=MODEL_CONFIGS[model_name],
            device=self.device,
            num_gpus=1,  # Only use 1 GPU per encoder instance
            batch_size=MODEL_CONFIGS[model_name]["batch_size"],
            use_default_instruction=use_default_instruction,
            use_fp16=use_fp16,
            testing_mode=testing_mode,
        )

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize model on the specific GPU."""
        home_dir = os.path.expanduser("~")
        model_path = os.path.join(
            home_dir, ".cache", "instructlab", "models", self.cfg.model_name
        )

        # In testing mode, allow direct download from HuggingFace
        if hasattr(self.cfg, "testing_mode") and self.cfg.testing_mode:
            logger.warning(
                f"Model not found locally at {model_path}. "
                "Testing mode enabled - downloading from HuggingFace..."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
            self.model = AutoModel.from_pretrained(
                self.cfg.model_name,
                add_pooling_layer=False,
                trust_remote_code=True,
            )
        else:
            if not os.path.exists(model_path):
                raise ValueError(
                    f"Model not found in available models: {self.cfg.model_name}\n"
                    "Please run `ilab model download` and download the necessary model"
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(
                model_path,
                add_pooling_layer=False,
                trust_remote_code=True,
                local_files_only=True,
            )

        if self.cfg.use_fp16:
            self.model = self.model.half()

        self.model = self.model.to(self.cfg.device)
        logger.info(f"Model loaded on device: {self.cfg.device}")

        # No need for DataParallel since we're running one encoder per GPU
        self.model.eval()

    def _prepare_inputs(
        self, texts: Union[str, List[str]], instruction: str = ""
    ) -> List[str]:
        """Prepare inputs with model-specific formatting."""
        if isinstance(texts, str):
            texts = [texts]

        # Ensure we always have an instruction
        if not instruction and not self.cfg.use_default_instruction:
            raise ValueError(
                "An instruction must be provided when use_default_instruction is False. "
                "Either provide an instruction or set use_default_instruction to True."
            )

        if (
            not instruction
            and self.cfg.use_default_instruction
            and self.cfg.model_config["default_instruction"]
        ):
            instruction = str(self.cfg.model_config["default_instruction"])

        if not instruction:  # catch if default_instruction is empty
            raise ValueError(
                "No instruction available. Either provide an instruction or ensure "
                "the model config has a valid default_instruction."
            )

        texts = [f"{instruction}: {text}" for text in texts]
        return texts

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str]],
        instruction: str = "",
        return_tensors: bool = True,
        show_progress: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encode texts into embeddings."""
        input_was_string = isinstance(inputs, str)
        inputs = self._prepare_inputs(inputs, instruction)

        encodings = self.tokenizer(
            inputs,
            max_length=self.cfg.model_config["max_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.cfg.device)

        embeddings_list = []
        for i in tqdm(
            range(0, len(inputs), self.cfg.batch_size),
            disable=not show_progress or len(inputs) < 256,
        ):
            batch = {k: v[i : i + self.cfg.batch_size] for k, v in encodings.items()}
            outputs = self.model(**batch)
            # Take the first token embedding (CLS) and normalize it
            embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
            embeddings_list.append(embeddings.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)
        if input_was_string:
            embeddings = embeddings[0]

        return embeddings if return_tensors else embeddings.numpy()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# FIXME: Use / Adapt below for unit / functional test for the encoder later
# def run_demo():
#     try:
#         encoder = ArcticEmbedEncoder(batch_size=2, max_length=512)
#         # Create some sample conversation texts. Multiply to have enough samples.
#         conversations = [
#             "User: I've been feeling really down lately...",
#             "User: I have a big presentation tomorrow...",
#             "User: I just read about the rapid decline in bee populations...",
#             "User: I'm planning a trip to Japan next year...",
#         ] * 10  # Adjust the number as needed

#         if encoder.cfg.rank == 0:
#             print("Last four conversations:")
#             print(conversations)

#         # Encode the texts using the encoder.encode method.
#         embeddings = encoder.encode(
#             conversations, instruction="Retrieve relevant passages."
#         )
#         if encoder.cfg.rank == 0:
#             print("\nEncode results:")
#             for i, (text, emb) in enumerate(zip(conversations, embeddings)):
#                 print(f"{i+1}. {text[:50]}... -> Embedding shape: {emb.shape}")

#         # Demonstrate using embed_dataset directly.
#         dataset = Dataset.from_dict(
#             {"text": conversations, "idx": list(range(len(conversations)))}
#         )
#         embedded_ds = encoder.embed_dataset(
#             dataset, instruction="Retrieve relevant passages.", add_to_dataset=True
#         )
#         if encoder.cfg.rank == 0:
#             print("\nDataset results:")
#             print(embedded_ds)

#         # Also show an example of returning numpy arrays.
#         embeddings_np = encoder.encode(
#             conversations,
#             instruction="Retrieve relevant passages.",
#             return_tensors=False,
#         )
#         if encoder.cfg.rank == 0:
#             print("\nNumpy array results:")
#             print(embeddings_np, embeddings_np.shape)
#     except Exception as e:
#         safe_print(dist.get_rank(), f"Demo failed: {str(e)}")
#     finally:
#         cleanup()


# if __name__ == "__main__":
#     run_demo()
