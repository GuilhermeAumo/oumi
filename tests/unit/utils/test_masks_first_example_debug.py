from pathlib import Path

import pytest

from oumi.utils.debug_utils import write_masks_first_example_debug_file


@pytest.mark.parametrize("with_labels", [False, True])
def test_write_masks_first_example_debug_file(tmp_path: Path, with_labels: bool):
    # Use a tiny, commonly-used tokenizer in this repo's tests.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    # Ensure PAD exists so attention/padding logic is stable.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode("Hello world", add_special_tokens=False)
    attention_mask = [1] * len(input_ids)
    labels = None
    if with_labels:
        # Mask out the first token; train on the rest.
        labels = [-100] + input_ids[1:]

    out_path = write_masks_first_example_debug_file(
        output_dir=tmp_path,
        raw_example={"some": "raw", "input_ids": input_ids},
        tokenizer=tokenizer,
        formatted_example_text="FORMATTED",
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        label_ignore_index=-100,
    )

    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "Original sample from dataset" in text
    assert "Formatted sample" in text
    assert "Tokenized sample" in text
    assert "Mask-separated formatted text" in text


def test_write_masks_first_example_debug_file_overwrites(tmp_path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode("Hello", add_special_tokens=False)

    out_path = write_masks_first_example_debug_file(
        output_dir=tmp_path,
        raw_example={"a": 1},
        tokenizer=tokenizer,
        formatted_example_text="A",
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=input_ids,
        label_ignore_index=-100,
    )
    first = out_path.read_text(encoding="utf-8")

    out_path2 = write_masks_first_example_debug_file(
        output_dir=tmp_path,
        raw_example={"b": 2},
        tokenizer=tokenizer,
        formatted_example_text="B",
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=input_ids,
        label_ignore_index=-100,
    )
    assert out_path2 == out_path
    second = out_path.read_text(encoding="utf-8")
    assert first != second
