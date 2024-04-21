import torch
from transformers import pipeline
import streamlit as st

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.bfloat16, device_map="auto")

st.title('Sam ChatBOT')
prompt = st.text_input('prompt', '')

if st.button('Envoyer'):
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True,
                   temperature=0, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"]
    st.write(response)
