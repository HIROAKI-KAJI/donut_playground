
from src.domain.receiptItem import ReceiptItem
from transformers import LogitsProcessor
from transformers import XLMRobertaTokenizer
from torch import Tensor, LongTensor, FloatTensor
from typing import cast


class InferenceLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: XLMRobertaTokenizer) -> None:
        self.tokenizer = tokenizer
        self.special_tokens = ReceiptItem.ReceiptItem.get_xml_tags()
        self.special_token_ids = cast(
            list[int],
            tokenizer.convert_tokens_to_ids(self.special_tokens),
        )
        self.tag_sequence = self.special_tokens  # ordered tag list

    def _last_tag(self, ids: Tensor) -> str:
        last_special_token_id = next(
            (token_id for token_id in reversed(ids.tolist()) if token_id in self.special_token_ids),
            None
        )
        return self.tokenizer.convert_ids_to_tokens(last_special_token_id) if last_special_token_id else "<s>"

    def _candidate_tags(self, last_tag: str) -> list[str]:
        try:
            idx = self.tag_sequence.index(last_tag)
            next_tag = self.tag_sequence[idx + 1]
            return [next_tag]
        except (ValueError, IndexError):
            return []  # No allowed next tag if not found or last tag

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        for i_row in range(len(input_ids)):
            ids = input_ids[i_row]
            last_tag_label = self._last_tag(ids)
            candidates = self._candidate_tags(last_tag_label)

            forbidden = [
                token_id
                for token_id in self.special_token_ids
                if self.tokenizer.convert_ids_to_tokens(token_id) not in candidates
            ]

            scores[i_row, forbidden] = -float("inf")

        return scores
