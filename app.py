import streamlit as st
from openai import OpenAI
from utils.doc_parser import parse_file
from utils.rag_utils import chunk_texts, build_vector_db, retrieve_top_k

# Get OpenAI API key from Streamlit secrets (for deployment) or fail safely
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("Please provide your OpenAI API key in Streamlit secrets.")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

    st.title("Supply Chain SOW Chat (RAG + GPT-4)")

    st.markdown("""
    **Instructions:**  
    - You can ask **any general supply chain, procurement, or SOW question** in the chat below—even without uploading files.
    - Upload your SOW Template as the **first file** below to use it as a reference.
    - Optionally, upload previous SOW examples or a draft SOW for review.
    - Then, use the chat box to get advice, feedback, or generate/review SOWs.
    """)

    sow_files = st.file_uploader(
        "Upload SOW Template (first) and Previous SOW Examples (.docx or .pdf, multiple allowed)",
        type=['docx', 'pdf'],
        accept_multiple_files=True,
        key="sow_files"
    )

    input_files = st.file_uploader(
        "Upload Input Documents (project info, requirements, or a draft SOW to review, etc.) (optional)",
        type=['docx', 'pdf'],
        accept_multiple_files=True,
        key="input_files"
    )

    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add new user message if entered
    prompt = st.chat_input("Ask a supply chain question, generate or review a SOW, or get advice...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display chat history (user and assistant)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Respond if last message is from user and not yet answered
    def needs_response():
        if len(st.session_state.messages) < 1:
            return False
        if st.session_state.messages[-1]["role"] != "user":
            return False
        if len(st.session_state.messages) >= 2 and st.session_state.messages[-2]["role"] == "assistant":
            return False
        return True

    if needs_response():
        prompt = st.session_state.messages[-1]["content"]

        # Prepare context from uploaded files
        if sow_files:
            sow_texts = [parse_file(f) for f in sow_files]
            template_text = sow_texts[0]
            examples_text = sow_texts[1:] if len(sow_texts) > 1 else []
        else:
            template_text = ""
            examples_text = []

        if input_files:
            input_content = "\n".join([parse_file(f) for f in input_files])
        else:
            input_content = ""

        if examples_text:
            example_docs = chunk_texts(examples_text)
            db = build_vector_db(example_docs, api_key=OPENAI_API_KEY)
            query = prompt
            if input_content:
                query += "\n" + input_content
            relevant_chunks = retrieve_top_k(db, query, k=5)
            retrieved_text = "\n---\n".join(relevant_chunks)
        else:
            retrieved_text = ""

        # <<<<<< Prompt that supports general and file-based Q&A >>>>>>
        prompt_for_llm = f"""You are an expert in supply chain management, SOW creation, and procurement best practices.

Here is the template for the SOW (if provided):
[TEMPLATE]
{template_text}

Here are relevant sections from previous SOWs (if provided):
[PREVIOUS_SOWS]
{retrieved_text}

Here is the draft SOW or input document (if provided):
[INPUT_SOW_OR_DOC]
{input_content}

User request:
{prompt}

Instructions:
- If the user asks for improvements, missing sections, or a review, carefully analyze [INPUT_SOW_OR_DOC] against the template and previous SOWs, and provide suggestions or highlight issues.
- If the user asks to generate a new SOW, do so as before.
- If the user asks a general question about supply chain management, procurement, or SOW best practices, answer helpfully and concisely.
- If no files are uploaded, you can still answer general questions or provide advice as a supply chain SOW expert.
- For general questions or advice, act as a friendly, expert assistant.
"""
        # <<<<<< End prompt >>>>>>

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert SOW writer and supply chain advisor."},
                        {"role": "user", "content": prompt_for_llm},
                    ],
                    max_tokens=2000,
                    temperature=0.2,
                )
                sow_text = response.choices[0].message.content.strip()
                st.markdown(sow_text)
                st.session_state.messages.append({"role": "assistant", "content": sow_text})

            st.download_button("Download Last SOW/Review/Advice", sow_text, file_name="generated_sow.txt", use_container_width=True)
