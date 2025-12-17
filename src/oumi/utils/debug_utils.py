# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Any, Optional, Union

import oumi.core.constants as constants
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.logging import logger


def log_example_for_debugging(
    raw_example: Any,
    formatted_example: str,
    tokenized_example: list[tuple[int, str]],
    model_input: dict[str, Any],
) -> None:
    """Logs an example of the data in each step for debugging purposes.

    Args:
        raw_example: The raw example from the dataset.
        formatted_example: The formatted example after processing.
        tokenized_example: The tokenized example after tokenization.
        model_input: The final model input after collating.
    """
    # Log to debug file
    logger.debug("Raw example: %s", raw_example)
    logger.debug("Formatted example: %s", formatted_example)
    logger.debug("Tokenized example: %s", tokenized_example)
    logger.debug("Model input: %s", model_input)


def _safe_json_dumps(obj: Any) -> str:
    """Best-effort JSON serialization for debug logs."""
    try:
        return json.dumps(
            obj, indent=2, ensure_ascii=False, default=str, sort_keys=True
        )
    except Exception:
        # Fallback to repr() if serialization fails (e.g., tensors, numpy arrays).
        return repr(obj)


def _as_list(x: Any) -> list[int]:
    """Converts tensors/arrays/scalars to a Python list of ints."""
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    if hasattr(x, "tolist"):
        val = x.tolist()
        if isinstance(val, list):
            # Flatten 1D only; higher dims are not expected here.
            return [int(v) for v in val]
        return [int(val)]
    # Scalar (int/np.int64/etc)
    return [int(x)]


def _decode_token(tokenizer: BaseTokenizer, token_id: int) -> str:
    # Decode token-by-token to preserve segmentation. Keep special tokens visible.
    try:
        return str(tokenizer.decode([token_id], skip_special_tokens=False))
    except Exception:
        # Extremely defensive; should not happen for HF tokenizers.
        return ""


def _render_mask_separated_text(
    *,
    tokens: list[tuple[int, str, bool, bool]],
) -> str:
    """Renders a token stream split into trained vs not-trained spans."""
    if not tokens:
        return ""

    def _header(is_trained: bool) -> str:
        return "---- Trained -----" if is_trained else "---- Not Trained -----"

    lines: list[str] = []
    current_flag: Optional[bool] = None
    current_text: str = ""

    for _tid, ttext, attn, train in tokens:
        # Treat padding (attn=False) as not-trained in the human view.
        is_trained = bool(attn) and bool(train)

        if current_flag is None:
            current_flag = is_trained
            current_text = ""

        if is_trained != current_flag:
            lines.append(_header(bool(current_flag)))
            lines.append("")
            lines.append(current_text)
            lines.append("")
            current_flag = is_trained
            current_text = ""

        current_text += ttext

    # Flush last segment.
    lines.append(_header(bool(current_flag)))
    lines.append("")
    lines.append(current_text)
    lines.append("")
    return "\n".join(lines)


def write_masks_first_example_debug_file(
    *,
    output_dir: Union[str, Path],
    raw_example: Any,
    tokenizer: BaseTokenizer,
    formatted_example_text: str,
    input_ids: Any,
    attention_mask: Any = None,
    labels: Any = None,
    label_ignore_index: Optional[int] = None,
    filename: str = "first_example_masks_debug.txt",
) -> Path:
    """Writes a detailed debug log for the first training example.

    The file is written to `output_dir/filename` and contains:
      1) Original sample from dataset
      2) Formatted sample text (after chat template), not tokenized
      3) Token list: (token id, decoded token, attention_mask bool, training_mask bool)
      4) A mask-separated rendering (trained vs not trained)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Convert ids/masks to 1D python lists.
    input_ids_list = _as_list(input_ids)
    if not input_ids_list:
        raise ValueError("Empty input_ids; cannot write mask debug file.")

    attention_list = _as_list(attention_mask)
    if not attention_list:
        attention_list = [1] * len(input_ids_list)
    if len(attention_list) != len(input_ids_list):
        # Best-effort: clamp to min length.
        min_len = min(len(attention_list), len(input_ids_list))
        input_ids_list = input_ids_list[:min_len]
        attention_list = attention_list[:min_len]

    labels_list = _as_list(labels)
    if labels_list and len(labels_list) != len(input_ids_list):
        min_len = min(len(labels_list), len(input_ids_list))
        input_ids_list = input_ids_list[:min_len]
        attention_list = attention_list[:min_len]
        labels_list = labels_list[:min_len]

    ignore_index = (
        int(label_ignore_index)
        if label_ignore_index is not None
        else int(constants.LABEL_IGNORE_INDEX)
    )

    token_tuples: list[tuple[int, str, bool, bool]] = []
    for i, tid in enumerate(input_ids_list):
        decoded = _decode_token(tokenizer, int(tid))
        attn = bool(attention_list[i]) if i < len(attention_list) else True
        train_mask = True
        if labels_list:
            train_mask = int(labels_list[i]) != ignore_index
        token_tuples.append((int(tid), decoded, attn, bool(train_mask)))

    mask_separated = _render_mask_separated_text(tokens=token_tuples)

    # Write as a single human-friendly text file.
    with out_path.open("w", encoding="utf-8") as f:
        f.write("## Oumi first-example masks debug log\n\n")

        f.write("### 1) Original sample from dataset\n")
        f.write(_safe_json_dumps(raw_example))
        f.write("\n\n")

        f.write("### 2) Formatted sample (chat template applied; not tokenized)\n")
        f.write(formatted_example_text)
        f.write("\n\n")

        f.write(
            "### 3) Tokenized sample (token_id, decoded_token, attention_mask, \
training_mask)\n"
        )
        # JSON-friendly representation.
        f.write(
            _safe_json_dumps(
                [
                    [tid, ttext, bool(attn), bool(train)]
                    for (tid, ttext, attn, train) in token_tuples
                ]
            )
        )
        f.write("\n\n")

        f.write("### 4) Mask-separated formatted text\n")
        f.write(mask_separated)
        f.write("\n")

    return out_path
