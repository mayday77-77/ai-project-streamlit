# utility.py
import streamlit as st
import hmac
from helper_functions.query import search_chunks


# login screen used in main page
def login_screen():
    st.title("🔐 Login")

    # create session state variables
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "login_success" not in st.session_state:
        st.session_state["login_success"] = False
    if "login_error" not in st.session_state:
        st.session_state["login_error"] = False

    # read valid credentials (from local secrets or Streamlit Cloud)
    gui_user = st.secrets["username"]
    gui_passwd = st.secrets["password"]   # stored in secrets

    # username input
    username = st.text_input("Username")

    # password input
    def password_entered():
        entered_password = st.session_state["password_input"]

        # secure comparison
        if username == gui_user and hmac.compare_digest(entered_password, gui_passwd):
            st.session_state["authenticated"] = True
            st.session_state["login_success"] = True
            st.session_state["login_error"] = False

        else:
            st.session_state["authenticated"] = False
            st.session_state["login_success"] = False
            st.session_state["login_error"] = True

        # remove password
        del st.session_state["password_input"]

    st.text_input(
        "Password",
        type="password",
        key="password_input",
        on_change=password_entered
    )

    # show login error if triggered
    if st.session_state.get("login_error"):
        st.error("❌ Invalid username or password.")

    # trigger rerun after callback is complete
    if st.session_state["login_success"]:
        st.session_state["login_success"] = False
        st.rerun()


# get policy options from vector DB
def get_policy_options():
    """ get distinct policy_ids from vectordb by getting metadata """
    try:
        hits = search_chunks("policy overview", _top_k=20, _policy_id=None)
        pids = sorted(
            { each_hit["metadata"].get("policy_id") for each_hit in hits if each_hit.get("metadata") }
        )
        return [each_pid for each_pid in pids if each_pid]
    except Exception as e:
        st.error(f"Error retrieving policy list: {e}")
        return []


# render source chunks in the UI
def render_sources(_sources):
    """ render retrieved source chunks in the UI """
    if not _sources:
        st.info("No sources retrieved.")
        return

    with st.expander("Sources (retrieved policy text)", expanded=False):
        for index, each_source in enumerate(_sources, start=1):
            meta = each_source.get("metadata", {})
            section = meta.get("section", "")
            subsection = meta.get("subsection", "")
            pages = f'{meta.get("page_start", "?")}-{meta.get("page_end", "?")}'
            fusion = each_source.get("fusion_score", None)

            header = f"{index}) {section} | {subsection} | pages {pages}"
            if fusion is not None:
                header += f" | score {fusion:.2f}"

            st.markdown(f"**{header}**")
            # cap display length
            st.code(each_source.get("text", "")[:3000])
            st.markdown("---")


# follow-up query expansion
def expand_followup_query(_current_query, _last_query):
    """
    merge the current user question with the last user question to provide some follow-up context without polluting the embedding query.
    only the previous question is used. (1 previous + current)
    """
    if not _last_query:
        return _current_query

    # If the follow-up question is too generic ("what about this?")
    # merging increases semantic meaning.
    merged = f"{_last_query} {_current_query}".strip()

    # limit embedding queries
    if len(merged) > 300:
        return _current_query

    return merged
