import with_sentence as ws
import google.generativeai as genai
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

genai.configure(api_key="")

def sentence_correction_AI(text):
    sentence = text
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Correct the sentence: {sentence}. Provide only the most probable corrected sentence."
    response = model.generate_content(prompt)
    propmt_output = response.text
    return propmt_output
