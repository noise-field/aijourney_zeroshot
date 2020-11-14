from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class ClassificationResults:
    raw_probabilities: Dict[str, float]
    normalized_probabilites: Dict[str, float]
    confident_enough: bool
    label: str
    error_message: Optional[str] = None


def move_dict_to_device(data: Dict[str, torch.Tensor], device: torch.device):
    """Moves all tensors in `data` to `device` inplace"""
    for key in data:
        data[key] = data[key].to(device)


class ZeroshotCategorizer:
    def __init__(self, model_path: str, max_input_length: int):
        self.max_input_length = max_input_length
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model = self.model.eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _get_label_mapping(self, labels: List[str]) -> Dict[str, int]:
        """Get indices of the first subtoken of each label
        
        Args:
            labels

        Returns:
            dict: mapping from labels to indices

        Raises:
            ValueError in case there are conflicting first subtokens
        """
        mapping = ({l: self.tokenizer(f" {l}")["input_ids"][0] for l in labels})
        return mapping

    def _preprocess_text(self, text: str, pattern: str, labels: List[str]) -> str:
        """Prepares text for feeding into the network
        
        Args:
            text:
            pattern:
            labels:
            
        Returns:
            TODO"""
        # TODO: enforce max input length
        if "{labels}" in pattern:
            return pattern.format(labels=labels, text=text)
        return pattern.format(text=text)

    def _get_label_probas(self, probas: Iterable[float], label_mapping: Dict[str, int]) -> Dict[str, int]:
        label_probas = dict()
        for label in label_mapping:
            label_probas[label] = probas[label_mapping[label]]
        return label_probas
        
    def _normalize_label_probas(self, label_probas: Dict[str, float]):
        label_probas_normalized = dict()
        probas_sum = sum(label_probas.values())

        for label in label_probas:
            label_probas_normalized[label] = label_probas[label] / probas_sum
        
        return label_probas_normalized

    def classify(
        self,
        text: str,
        pattern: str,
        labels: List[str],
        min_confidence_threshold: float,
        min_confidence_ratio: float
    ) -> ClassificationResults:
        """Classify the `text` into `labels` using zero-shot approach
        
        Args:
            text:
            pattern:
            labels:

        Returns:
            ClassificationResults: 
        """
        text = self._preprocess_text(text, pattern, labels)
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_input_length,
            return_attention_mask=False,
            return_tensors="pt",
        )

        tokenized = move_dict_to_device(tokenized)

        with torch.no_grad():
            res = self.model.forward(**tokenized, return_dict=True)

        logits = res["logits"][0, -1]
        order = logits.argsort(descending=True)
        probas = torch.softmax(logits, 0).cpu().detach().numpy()

        label_mapping = self._get_label_mapping(labels)
        label_probas = self._get_label_probas(probas, label_mapping)
        normalized_probas = self._normalize_label_probas(label_probas)

