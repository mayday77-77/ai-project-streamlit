# Home.py: streamlit app for the Insurance Policy Assistant RAG system.

import streamlit as st
from helper_functions.vectordb import list_policy_ids
from helper_functions.query import answer_question, search_chunks
from helper_functions.utility import expand_followup_query, render_sources, login_screen

# set header config
st.set_page_config(page_title="Policy Assistant", page_icon="💬", layout="wide")

# authentication UI
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# If not authenticated: show login
if not st.session_state.get("authenticated", False):
    login_screen()
    st.stop()

# logout button in sidebar
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# sidebar settings
st.sidebar.title("⚙️ Settings")

# disclaimer for project
with st.expander("⚠️ Disclaimer ⚠️", expanded=False):
    st.markdown("**IMPORTANT NOTICE**")
    st.markdown('''This web application is a prototype developed for :orange[educational purposes only].
                The information provided here is :orange[NOT intended for real-world usage] and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.  
                :orange[Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.]  
                Always consult with qualified professionals for accurate and personalized advice.''')

policy_list = list_policy_ids()
policy_options = policy_list
selected_policy = st.sidebar.selectbox("Select policy", policy_options, index=0)

# need to reset chat if policy changes to remove all previous context
if "last_policy" not in st.session_state:
    st.session_state["last_policy"] = selected_policy

if selected_policy != st.session_state["last_policy"]:
    st.session_state["last_policy"] = selected_policy
    st.session_state["messages"] = []
    st.session_state["last_user_question"] = None

# default to 7
top_k = st.sidebar.slider("Top chunks to use (top_k)", min_value=3, max_value=10, value=7, step=1)

show_sources = st.sidebar.checkbox("Show sources", value=False)

# clear chat button
if st.sidebar.button("🧹 Clear chat"):
    st.session_state.messages = []
    st.session_state.last_user_question = None
    # to show info message
    st.session_state["clear_chat_flag"] = True
    st.rerun()

# show success message after clearing chat
if st.session_state.get("clear_chat_flag", False):
    st.sidebar.info("🧹 Chat history has been cleared.")
    # remove the flag so it only shows once
    st.session_state.clear_chat_flag = False


# session state init
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = None

# main page UI
st.title("💬 Insurance Policy Assistant")
st.caption("Get answers to your insurance questions, backed by official policy documents.")

# render past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and show_sources:
            render_sources(message.get("sources", []))

# chat input
user_query = st.chat_input("➡️ Type your question here…")

if user_query:
    # follow-up query expansion
    expanded_query = expand_followup_query(
        _current_query=user_query,
        _last_query=st.session_state.last_user_question
    )

    # update memory with latest user question
    st.session_state.last_user_question = user_query

    # log user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # start the AI chat and retrieval
    policy_id = selected_policy

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                retrieved = search_chunks(expanded_query, _top_k=top_k, _policy_id=policy_id)
                result = answer_question(
                    _question=user_query,
                    _policy_id=policy_id,
                    _top_k=top_k,
                    _pre_retrieved_chunks=retrieved,
                )
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)
                if show_sources:
                    render_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                err_msg = f"Error: `{e}`"
                st.error(err_msg)
