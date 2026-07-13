"""
app.py
-------------------------------------------------

Streamlit UI for Private Agentic RAG System

Features:

1. PDF Upload
2. Build Knowledge Index
3. Core vs Agentic RAG selection
4. Chat Interface
5. Agentic Metrics Display

"""


"""
app.py
-------------------------------------------------

Streamlit UI for Private Agentic RAG System
"""


import os

import sys


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


if PROJECT_ROOT not in sys.path:

    sys.path.append(
        PROJECT_ROOT
    )


import streamlit as st


from application.ui_service.rag_service import (

    list_documents,

    upload_document,

    delete_documents,

    delete_all_documents,

    build_knowledge_base,

    run_core_rag,

    run_agentic_rag

)



# --------------------------------------------
# Page Configuration
# --------------------------------------------


st.set_page_config(

    page_title="Private Agentic RAG",

    layout="wide"

)



st.title(

    "Private Self-Correcting Agentic RAG System"

)



st.markdown(

    "Privacy Preserving RAG using FAISS, SLM, ReRanking and Agentic Self Correction"

)



# --------------------------------------------
# Sidebar
# --------------------------------------------


# --------------------------------------------
# Sidebar
# --------------------------------------------

st.sidebar.title("📚 Knowledge Base")

documents = list_documents()

if len(documents) == 0:

    st.sidebar.info(
        "No PDF documents uploaded."
    )

else:

    st.sidebar.subheader(
        "Existing Documents"
    )

    selected_documents = []

    for doc in documents:

        status = "🟢 Indexed" if doc["indexed"] else "🟡 Needs Build"

        checked = st.sidebar.checkbox(

            f"{doc['filename']} ({status})",

            key=doc["filename"]

        )

        if checked:

            selected_documents.append(

                doc["filename"]

            )


st.sidebar.divider()


uploaded_file = st.sidebar.file_uploader(

    "Upload PDF",

    type=["pdf"]

)


if uploaded_file:

    upload_document(

        uploaded_file

    )

    st.sidebar.success(

        "Document uploaded successfully."

    )

    st.sidebar.info(

        "Knowledge Base changed.\n\nClick Build Knowledge Base."

    )


col1, col2 = st.sidebar.columns(2)

with col1:

    if st.button(

        "🗑 Delete"

    ):

        if len(selected_documents) > 0:

            with st.spinner(

                "Updating Knowledge Base..."

            ):

                result = delete_documents(

                    selected_documents

                )

            st.success(

                f"{result['deleted']} document(s) deleted."

            )

            st.success(

                result["status"]

            )

            if result["rebuilt"]:

                st.caption(

                    "✅ Vector Database synchronized successfully."

                )

            else:

                st.caption(

                    "📂 Knowledge Base is empty."

                )

            st.rerun()


with col2:

    if st.button(

        "🗑 Delete All"

    ):

        with st.spinner(

            "Updating Knowledge Base..."

        ):

            result = delete_all_documents()

        st.success(

            result["status"]

        )

        if result["rebuilt"]:

            st.caption(

                "✅ Vector Database synchronized successfully."

            )

        else:

            st.caption(

                "📂 Knowledge Base is empty."

            )

        st.rerun()


st.sidebar.divider()


if st.sidebar.button(

    "🛠 Build Knowledge Base",

    use_container_width=True

):

    with st.spinner(

        "Building Knowledge Base..."

    ):

        result = build_knowledge_base()

    st.sidebar.success(

        result["status"]

    )

    st.sidebar.write(

        "Documents:",

        result["documents"]

    )

    st.sidebar.write(

        "Chunks:",

        result["chunks"]

    )


st.sidebar.divider()


pipeline = st.sidebar.radio(

    "Pipeline",

    [

        "Core RAG",

        "Agentic RAG"

    ]

)



# --------------------------------------------
# Chat Section
# --------------------------------------------


st.divider()



question = st.chat_input(

    "Ask a question from your documents"

)



if question:


    st.chat_message(

        "user"

    ).write(

        question

    )



    with st.spinner(

        "Generating answer..."

    ):


        if pipeline == "Core RAG":


            result = run_core_rag(

                question

            )


        else:


            result = run_agentic_rag(

                question

            )



    st.chat_message(

        "assistant"

    ).write(

        result["answer"]

    )



    # ----------------------------------------
    # Agentic Details
    # ----------------------------------------


    if result["mode"] == "Agentic RAG":


        st.divider()


        st.subheader(

            "Agentic Reasoning"

        )



        col1, col2, col3 = st.columns(

            3

        )



        col1.metric(

            "Confidence",

            result["reflection"][

                "confidence_score"

            ]

        )



        col2.metric(

            "Similarity",

            result["reflection"][

                "similarity_score"

            ]

        )



        col3.metric(

            "Retries",

            result[

                "retry_count"

            ]

        )



        st.write(

            "Self Corrected:",

            result[

                "self_corrected"

            ]

        )



        st.write(

            "Planner Decision"

        )



        st.json(

            result["plan"]

        )



    # ----------------------------------------
    # Sources
    # ----------------------------------------


    with st.expander(

        "Retrieved Context"

    ):


        if "context_chunks" in result:


            for item in result[

                "context_chunks"

            ]:


                st.write(

                    item

                )


        elif "context" in result:


            for item in result[

                "context"

            ]:


                st.write(

                    item

                )
