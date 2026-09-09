import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖"
)

st.title("🤖 AI Text Generator")
st.write("✨ Enter a sentence and let AI complete it!")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-125M"
    )

generator = load_model()

prompt = st.text_area(
    "✍️ Enter your text:",
    placeholder="Artificial Intelligence is..."
)

if st.button("✨ Generate Text"):

    if prompt.strip():

        with st.spinner("🤖 Generating..."):

            result = generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        generated_text = result[0]["generated_text"]

        # Remove the original prompt
        new_text = generated_text[len(prompt):].strip()

        st.subheader("📝 Generated Text")

        if new_text:
            st.write(new_text)
        else:
            st.warning("No additional text was generated.")

    else:
        st.warning("⚠️ Please enter some text first!")