import streamlit as st

st.set_page_config(layout="wide", page_title="About This App")

st.title("ℹ️ About This App")
st.markdown("""
This is a capstone project assignment for the :blue-background[AI bootcamp course] to demonstrate building a Retrieval-Augmented Generation (RAG) application using Streamlit and OpenAI's LLMs.

The idea behind this project is to attempt to help users better understand :blue-background[Insurance policy] documents by leveraging AI to provide clear, accurate answers grounded in the actual policy text.

This app processes multiple policy PDFs, extracts structured information, and uses AI to answer your
questions in a clear and friendly way.
""")
st.markdown("---")

# project scope
st.header("💡Project Scope")
st.subheader("Problem Statement")
st.markdown("""
Insurance policy documents are very lengthy, filled with legal and technical jargon, and difficult for policy holders to quickly interpret and understand the full coverage.
We as customers do struggle to find specific information such as coverage limits, exclusions, or claim procedures within the policy.

Although there are Insurance agents who can assist, they themselves might miss certain specifics sometimes and may cause confusion.

Many disputes and claims arise because policy holders misunderstand coverage, exclusions, or claim conditions due to the complexity of the policy.
""")
st.subheader("Proposed Solution")
st.markdown("""
An :green[AI assistant chatbot] on official policy documents could be handy and it can instantly provide accurate answers, serving as an internal knowledge companion.
""")

st.subheader("Impact")
st.markdown("""
Having an :green[AI assistant chatbot] that can help answer questions quickly about insurance policies enables faster understanding, more accurate responses, and higher overall satisfaction for policy holders.
""")
st.markdown("---")

# objective
st.header("✅ Objective")
st.markdown("""
To take what we have learned in the bootcamp and attempt to build a functional :green[Web application] that can:

- 📄 Read and preprocess some sample publicly available insurance policy PDFs.  
- 🧩 Break documents into meaningful chunks.
- 📦 Store them in a local **Chroma** vector database. 
- 🔍 Use embeddings to retrieve the most relevant information.  
- 🎯 Apply a reranking layer to boost accuracy.
- 💬 Generate answers using an LLM with controlled prompting.  

All this is to help ensure that the LLM responses can stay truthful to the original policy documents or trying to coax out more specific answers.
""")
st.markdown("---")

# datasource
st.header("📊 Data Sources")
st.markdown("""
In order to try to build a functional application, some sample policy documents are needed to be pre-processed and used as the knowledge base.

There are publicly available sample insurance policy documents from one insurance provider and they will be used as the data source for this project.
>Reference: [Document policies site](https://www.income.com.sg/policy-documents-and-forms)

The number of policies are vast, thus for this small project, I will just be covering 5 different types of policies:
1. [Cancer Care Policy](https://www.income.com.sg/kcassets/02b27fba-34d1-4e70-be92-350496ad3477/Policy%20Contract%20-%20Complete%20Cancer%20Care%20%28CBN3%29.pdf)
2. [Lady 360 Policy](https://www.income.com.sg/kcassets/61bc2a57-a239-4730-bc7f-580c0ed67a25/Policy%20Contract%20-%20Lady%20360%20%28LPN%29.pdf)
3. [Life Secure Policy](https://www.income.com.sg/kcassets/5c765a38-627b-49cc-abe1-1d36f7d40564/Policy%20Contract%20-%20Complete%20Life%20Secure%20%28VQMW%20_%20VQVW%29.pdf)
4. [Private Car Policy](https://www.income.com.sg/kcassets/f1c997ad-f070-412c-8540-d5d6cba36a7f/GPC%20Prestige%20T%26C%20Website%20--%20012024.pdf)
5. [Silver Secure Policy](https://www.income.com.sg/kcassets/e3c99391-b0de-437f-8d54-de0430ce3109/20180301-Policy-Conditions-Silver-Secure-%28SHN%29.pdf)

The documents could also be previewed and downloaded from the :blue[[View All Policies](View_All_Policies)] page.
""")
st.markdown("---")

# features
st.header("📊 Features")
st.markdown("""
The project aims to implement the following features:
- ⚠️ :green[Disclaimer] notice about the webpage's purpose.
- 📦 Use of persistent :green[vector database (Chroma)] to store and retrieve document embeddings.
- 🤖 A user-friendly Streamlit :green[Chatbot interface] for questions and answers based on the preloaded policy documents.
- 🗳️ Ability to select specific policy document to query.
- 👤 :green[Login prompt] for username and password authentication as well as :green[Logout button].
- 🦾 Use of OpenAI's :green[LLM (GPT-4o-mini)] for generating relevant answers.
- 📄 :green[Methodology page] explaining the technical approach and architecture.
""")
st.markdown("---")

# Sample screenshots
st.header("📸 Sample")
st.write(""":blue[Login Page]""")
st.image("pages/images/login.png")
st.write(""":blue[Chat Interface]""")
st.image("pages/images/chat.png")
