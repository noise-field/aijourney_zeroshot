"""
Fetch ruGPT3 models and save them for further offline use.
"""
import argparse
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.DEBUG)


def fetch_and_save(model_name: str, output_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logging.info("Model fetched")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("Tokenizer fetched")
    model.save_pretrained(output_path)
    logging.info("Model saved")
    tokenizer.save_pretrained(output_path)
    logging.info("Tokenizer saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch ruGPT3 models and save them for further offline use."
    )
    parser.add_argument("model_name")
    parser.add_argument("output_path")

    args = parser.parse_args()

    fetch_and_save(args.model_name, args.output_path)
