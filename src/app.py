"""A streamlit app to showcase ruGPT3-based zero-shot classification"""
import streamlit as st

from classifier import ZeroshotClassifier

MODEL_PATH = "./model"
MAX_LENGTH = 1024

DEFAULT_PROMPT = "Текст: {text}\nКатегория: "
DEFAULT_MIN_THRESHOLD = 0.001
DEFAULT_MIN_RATIO = 1.5

@st.cache(allow_output_mutation=True)
def load_classifier():
    return ZeroshotClassifier(MODEL_PATH, MAX_LENGTH)

classifier = load_classifier()

st.title("Категоризация текстов на основе ruGPT-3 без дообучения")

prompt_pattern = st.sidebar.text_area("Паттерн запроса к модели", value=DEFAULT_PROMPT)
confidence_threshold = st.sidebar.number_input("Минимальный порог вероятности", value=DEFAULT_MIN_THRESHOLD, step=0.0001, format="%.4f", key="cfth")
confidence_ratio = st.sidebar.number_input("Минимальное отношение вероятностей", value=DEFAULT_MIN_RATIO, step=.1, key="cfr")

input_text = st.text_area("Введите текст для классификации")
labels_input = st.text_input("Введите метки через запятую")

labels = labels_input.split(",")

if st.button("Классифицировать"):
    clf_result = classifier.classify(
        input_text, prompt_pattern, labels, confidence_threshold, confidence_ratio
    )
    if clf_result.confident_enough:
        st.write("*ruGPT3:* я думаю, что этот текст относится к категории _{}_".format(clf_result.label))
    else:
        st.write("*ruGPT3:* я не знаю точно, к чему отнести этот текст")
