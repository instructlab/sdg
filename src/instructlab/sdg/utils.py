# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Optional, Sequence, Union
import copy
import dataclasses
import io
import json
import logging
import math
import os
import sys

# Third Party
# instructlab - TODO these need to go away, issue #6
from instructlab.configuration import DEFAULT_API_KEY, DEFAULT_MODEL_OLD
from instructlab.utils import get_sysprompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, OpenAIError
import httpx

StrOrOpenAIObject = Union[str, object]

DEFAULT_CHUNK_OVERLAP = 100


class GenerateException(Exception):
    """An exception raised during generate step."""


# pylint: disable=too-many-instance-attributes
@dataclasses.dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: Optional[int] = None


def openai_completion(
    api_base,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="ggml-merlinite-7b-lab-Q4_K_M",
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    api_key=DEFAULT_API_KEY,
    **decoding_kwargs,
) -> Union[
    Union[StrOrOpenAIObject],
    Sequence[StrOrOpenAIObject],
    Sequence[Sequence[StrOrOpenAIObject]],
]:
    """Decode with OpenAI API.

    Args:
        api_base: Endpoint URL where model is hosted
        tls_insecure: Disable TLS verification
        tls_client_cert: Path to the TLS client certificate to use
        tls_client_key: Path to the TLS client key to use
        tls_client_passwd: TLS client certificate password
        prompts: A string or a list of strings to complete. If it is a chat model the strings
            should be formatted as explained here:
            https://github.com/openai/openai-python/blob/main/chatml.md.
            If it is a chat model it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This will be deprecated in the future.
        return_text: If True, return text instead of full completion object (e.g. includes logprob).
        api_key: API key API key for API endpoint where model is hosted
        decoding_kwargs: Extra decoding arguments. Pass in `best_of` and `logit_bias` if needed.

    Returns:
        A completion or a list of completions. Depending on return_text, return_openai_object,
        and decoding_args.n, the completion type can be one of:
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in enumerate(prompt_batches):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        shared_kwargs = {
            "model": model_name,
            **batch_decoding_args.__dict__,
            **decoding_kwargs,
        }

        if not api_key:
            # we need to explicitly set non-empty api-key, to ensure generate
            # connects to our local server
            api_key = "no_api_key"

        # do not pass a lower timeout to this client since generating a dataset takes some time
        # pylint: disable=R0801
        orig_cert = (tls_client_cert, tls_client_key, tls_client_passwd)
        cert = tuple(item for item in orig_cert if item)
        verify = not tls_insecure
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            http_client=httpx.Client(cert=cert, verify=verify),
        )

        # ensure the model specified exists on the server. with backends like vllm, this is crucial.
        model_list = client.models.list().data
        model_ids = []
        for model in model_list:
            model_ids.append(model.id)
        if not any(model_name == m for m in model_ids):
            if model_name == DEFAULT_MODEL_OLD:
                logging.info(
                    "Model %s is not a full path. Try running ilab init or edit your config to have the full model path for serving, chatting, and generation.",
                    model_name,
                )
            raise GenerateException(
                f"Model {model_name} is not served by the server. These are the served models {model_ids}"
            )

        messages = [
            {"role": "system", "content": get_sysprompt()},
            {"role": "user", "content": prompt_batch[batch_id]},
        ]

        # Inference the model
        try:
            response = client.chat.completions.create(
                messages=messages,
                **shared_kwargs,
            )
        except OpenAIError as exc:
            raise GenerateException(
                f"There was a problem connecting to the server {exc}"
            ) from exc

        completions.extend(response.choices)

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make a nested list, where each entry is consecutive decoding_args.n of original entries.
        completions = [
            completions[i : i + decoding_args.n]
            for i in range(0, len(completions), decoding_args.n)
        ]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    # pylint: disable=consider-using-with
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding="utf-8")
    return f


def _make_r_io_base(f, mode: str):
    # pylint: disable=consider-using-with
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding="utf-8")
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    with _make_w_io_base(f, mode) as f_:
        if isinstance(obj, (dict, list)):
            json.dump(obj, f_, indent=indent, default=default)
        elif isinstance(obj, str):
            f_.write(obj)
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    with _make_r_io_base(f, mode) as f_:
        return json.load(f_)


def num_tokens_from_words(num_words) -> int:
    return int(num_words * 1.3)  # 1 word ~ 1.3 token


def num_chars_from_tokens(num_tokens) -> int:
    return int(num_tokens * 4)  # 1 token ~ 4 English character


def num_tokens_from_chars(num_chars) -> int:
    return int(num_chars / 4)  # 1 token ~ 4 English character


def max_seed_example_tokens(server_ctx_size, prompt_num_chars) -> int:
    """
    Estimates the maximum number of tokens any seed example can have based
    on the server context size and number of characters in the selected prompt.

    A lot has to fit into the given server context size:
      - The prompt itself, which can vary in size a bit based on model family and knowledge vs skill
      - Two seed examples, which we append to the prompt template.
      - A knowledge document chunk, if this is a knowledge example.
      - The generated completion, which can vary substantially in length.

    This is an attempt to roughly estimate the maximum size any seed example
    (question + answer + context values from the yaml) should be to even have
    a hope of not often exceeding the server's maximum context size.

    NOTE: This does not take into account knowledge document chunks. It's meant
    to calculate the maximum size that any seed example should be, whether knowledge
    or skill. Knowledge seed examples will want to stay well below this limit.

    NOTE: This is a very simplistic calculation, and examples with lots of numbers
    or punctuation may have quite a different token count than the estimates here,
    depending on the model (and thus tokenizer) in use. That's ok, as it's only
    meant to be a rough estimate.

    Args:
        server_ctx_size (int): Size of the server context, in tokens.
        prompt_num_chars (int): Number of characters in the prompt (not including the examples)
    """
    # Ensure we have at least 1024 tokens available for a response.
    max_seed_tokens = server_ctx_size - 1024
    # Subtract the number of tokens in our prompt template
    max_seed_tokens = max_seed_tokens - num_tokens_from_chars(prompt_num_chars)
    # Divide number of characters by 2, since we insert 2 examples
    max_seed_tokens = int(max_seed_tokens / 2)
    return max_seed_tokens


def chunk_document(documents: List, server_ctx_size, chunk_word_count) -> List[str]:
    """
    Iterates over the documents and splits them into chunks based on the word count provided by the user.
    Args:
        documents (dict): List of documents retrieved from git (can also consist of a single document).
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
    Returns:
         List[str]: List of chunked documents.
    """
    no_tokens_per_doc = num_tokens_from_words(chunk_word_count)
    if no_tokens_per_doc > int(server_ctx_size - 1024):
        raise ValueError(
            "Error: {}".format(
                str(
                    f"Given word count ({chunk_word_count}) per doc will exceed the server context window size ({server_ctx_size})"
                )
            )
        )
    content = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=num_chars_from_tokens(no_tokens_per_doc),
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    for docs in documents:
        temp = text_splitter.create_documents([docs])
        content.extend([item.page_content for item in temp])

    return content
