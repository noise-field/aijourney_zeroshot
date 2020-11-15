from dataclasses import dataclass
from typing import List, Dict

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# A lazy way to account for special tokens and other unexpected
# factors when stripping input text to a maximum desired
# number of subword units
MAX_LENGTH_FUDGE_FACTOR = 15


@dataclass
class ClassificationResults:
    raw_probabilities: Dict[str, float]
    normalized_probabilites: Dict[str, float]
    confident_enough: bool
    label: str


class ZeroshotClassifier:
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

    def _preprocess_text(self, text: str, prompt_pattern: str, labels: List[str]) -> str:
        """Prepares text for feeding into the network
        
        Args:
            text: text to be classified
            prompt_pattern: text to elicit classification response from the model.
                Should contain `{text}` placeholder.
            labels: list of labels to classify into
            
        Returns:
            str: text ready to be fed into the model
        """
        # calculate the length without format placeholders
        if "{text}" not in prompt_pattern:
            raise ValueError("The prompt must contain `{text}` placeholder")

        stripped_pattern = prompt_pattern.replace("{text}", "").replace("{labels}", "")
        pattern_length = len(self.tokenizer(stripped_pattern)["input_ids"])
        max_text_length = self.max_input_length - pattern_length

        if "{labels}" in prompt_pattern:
            labels_length = len(self.tokenizer(", ".join(labels))["input_ids"])
            max_text_length -= labels_length

        max_text_length -= MAX_LENGTH_FUDGE_FACTOR

        # cut the input text so that it fits into `max_length` together with the prompt
        text = self.tokenizer.decode(self.tokenizer(text)["input_ids"][:max_text_length])

        if "{labels}" in prompt_pattern:
            return prompt_pattern.format(labels=", ".join(labels), text=text)
        return prompt_pattern.format(text=text)

    @staticmethod
    def _get_label_probas(probas: List[float], label_mapping: Dict[str, int]) -> Dict[str, float]:
        """Get label probabilities
        
        Get probabilities of the tokens associated with the first 
        subtokens of the labels from the model output"""
        label_probas = dict()
        for label in label_mapping:
            label_probas[label] = probas[label_mapping[label]]
        return label_probas
        
    @staticmethod
    def _normalize_label_probas(label_probas: Dict[str, float]) -> Dict[str, float]:
        """Get unmathematical probabilities for labels as if other tokens do not exist"""
        label_probas_normalized = dict()
        probas_sum = sum(label_probas.values())

        for label in label_probas:
            label_probas_normalized[label] = label_probas[label] / probas_sum
        
        return label_probas_normalized

    def _get_next_token_probabilites(self, text: str) -> List[float]:
        """Perform the forward pass on the model and return the softmax"""
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_input_length,
            return_attention_mask=False,
            return_tensors="pt",
        )

        tokenized = tokenized.to(self.device)

        with torch.no_grad():
            res = self.model.forward(**tokenized, return_dict=True)

        logits = res["logits"][0, -1]
        probas = torch.softmax(logits, 0).cpu().detach().numpy()
        return probas

    def classify(
        self,
        text: str,
        prompt_pattern: str,
        labels: List[str],
        min_confidence_threshold: float=0.001,
        min_confidence_ratio: float=1.5
    ) -> ClassificationResults:
        """Classify the `text` into `labels` using zero-shot approach
        
        Args:
            text: text to be classified
            prompt_pattern: text to elicit classification response from the model.
                Should contain `{text}` placeholder.
            labels: list of labels to classify into

        Returns:
            ClassificationResults: results of zeroshot prediction
        """
        if len(labels) < 2:
            raise ValueError("Please, provide at least two labels.")

        text = self._preprocess_text(text, prompt_pattern, labels)
        probas = self._get_next_token_probabilites(text)

        label_mapping = self._get_label_mapping(labels)
        label_probas = self._get_label_probas(probas, label_mapping)
        normalized_probas = self._normalize_label_probas(label_probas)

        labels_sorted = sorted(label_probas.items(), key=lambda x: x[1], reverse=True)

        most_probable_label = labels_sorted[0][0]

        absolute_probabilty = labels_sorted[0][1]
        probability_ratio = labels_sorted[0][1] / labels_sorted[1][1]

        confident_enough = (absolute_probabilty > min_confidence_threshold) and \
           (probability_ratio > min_confidence_ratio)

        return ClassificationResults(
            label_probas,
            normalized_probas,
            confident_enough,
            most_probable_label
        )
