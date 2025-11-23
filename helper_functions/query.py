# query.py
from helper_functions.vectordb import embed_texts, load_existing_collection
from helper_functions.llm import openai_client
from rank_bm25 import BM25Okapi
import re

# Global vars
EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "policies"
PERSIST_DIR = "vectordb"
LLM_MODEL = "gpt-4o-mini"
CANDIDATE_MULTIPLIER_DEFAULT = 8
CANDIDATE_MULTIPLIER_DEFINITION = 20
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def refine_query(_question):
    """ Refining user query by removing conversational words before embedding """
    query = _question.strip().lower()

    # words that can be removed from query
    filter_words = [
         "can you", "could you", "explain", "explain to me", "give me",
        "hey", "hi", "hello", "help", "help me", "i want to know",
        "pls", "please", "tell me"
    ]
    for word in filter_words:
        query = query.replace(word, "")

    query = query.strip()
    # add context to help embedding model retrieve better
    refined = (
        query.strip() +
        "\n\n[context: insurance policy, benefits, coverage, exclusions, "
        "waiting period, definitions, payouts, claims, conditions]"
    )
    if is_definition_query(_question):
        refined += "\nLook for official policy definitions / criteria."

    return refined


def get_embedding(_question):
    """ Get embedding vector for user question after refining """
    clean_query = refine_query(_question)
    vector = embed_texts([clean_query], _model=EMBED_MODEL)

    return vector[0]


def is_definition_query(_question):
    """ Heuristic to determine if the query is asking for definitions / criteria """
    return any(keyword in _question.lower() for keyword in [
        "definition", "define", "meaning", "criteria",
        "stage", "staging", "classified", "diagnostic"
    ])


def minmax_norm(_values):
    """ Min-max normalization of a list of values to [0, 1] """
    if not _values:
        return _values
    vmin, vmax = min(_values), max(_values)
    if vmax - vmin < 1e-9:
        return [0.0 for _ in _values]

    return [(value - vmin) / (vmax - vmin) for value in _values]


def search_chunks(_query, _top_k=5, _policy_id=None):
    """ Search chunks in vectordb and return top_k results with reranker """
    collection = load_existing_collection(
        _persist_dir=PERSIST_DIR,
        _collection_name=COLLECTION_NAME
    )

    # get embedding query
    query_embed = get_embedding(_query)

    policy_indicator = {}
    if _policy_id:
        policy_indicator["policy_id"] = _policy_id

    # determine candidate multiplier based on query type
    multiplier = CANDIDATE_MULTIPLIER_DEFINITION if is_definition_query(_query) else CANDIDATE_MULTIPLIER_DEFAULT

    num_candidates = _top_k * multiplier

    result = collection.query(
        query_embeddings=[query_embed],
        n_results=num_candidates,
        where=policy_indicator
    )

    # chunk info
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    candidates = []
    for doc, meta, dist in zip(docs, metas, dists):
        candidates.append({
            "text": doc,
            "metadata": meta,
            "distance": float(dist),
        })

    if not candidates:
        return []

    # convert distance to similarity-like score
    # similarity = 1 / (1 + distance)
    dense_scores = [1.0 / (1.0 + each_candidate["distance"]) for each_candidate in candidates]
    dense_scores_norm = minmax_norm(dense_scores)

    # rerank with BM25 scores
    bm25 = BM25Okapi([
        # tokenizer for BM25, keeps alphanumerics, splits on non-letters
        re.findall(r"[a-z0-9]+", each_candidate["text"].lower())
        for each_candidate in candidates])
    bm25_scores_norm = minmax_norm(
        bm25.get_scores(
            re.findall(r"[a-z0-9]+", _query.lower())).tolist()
        )

    # fusion scoring
    for index, candidate in enumerate(candidates):
        fusion = (DENSE_WEIGHT * dense_scores_norm[index] + BM25_WEIGHT * bm25_scores_norm[index])
        candidate["dense_score"] = dense_scores_norm[index]
        candidate["bm25_score"] = bm25_scores_norm[index]
        candidate["fusion_score"] = fusion

    candidates.sort(key=lambda x: x["fusion_score"], reverse=True)
    # return top_k fused chunks
    return candidates[:_top_k]


SYSTEM_PROMPT = """
You are a friendly, helpful, and reliable insurance policy assistant.

Your responsibilities:
1. Always answer using ONLY the information found in the provided context. Never guess or invent policy details.
2. If the context fully answers the user’s question:
   - Provide a clear explanation first.
   - Then provide helpful elaboration if the topic is complex (such as definitions, diagnostic criteria, or benefit conditions).
   - Use short paragraphs and bullet points to make the answer easy to read.

3. If the context partially answers the question:
   - Provide the partial information clearly.
   - Explain what is included and what is not included.
   - Add a friendly suggestion such as:
     "For the complete details, you may want to check the full policy document or contact our customer service team."

4. If the context does NOT include the answer:
   - Do NOT say the information does not exist.
   - Do NOT guess.
   - Reply politely with something like:
     "I checked the policy text provided, but it does not mention this specific detail.
      For further clarification, you may want to contact our customer service team or your insurance representative.
      I’m here if you’d like help exploring a related section of the policy."

5. When referencing a section, page number, or subsection:
   - Do NOT simply tell the user to 'refer to section X'.
   - Instead, provide a useful, meaningful summary of the content from that section.
   - If the section contains definitions, describe the key parts.
   - If it contains diagnostic criteria, summarise the criteria.
   - If the retrieved chunks include multiple related passages, combine and summarize them clearly.

6. If the question is broad:
   - Provide a short summary first.
   - Then give more detail using bullet points.
   - Offer to narrow down the answer if the user has a more specific follow-up.

7. Format guidelines:
   - Use bullet points for lists.
   - Use short paragraphs.
   - Highlight key terms when helpful.
   - Keep explanations simple and customer-friendly.
   - Avoid medical or legal jargon unless it appears in the policy text itself.

8. After you provide the main explanation or summary, always add a short, friendly follow-up line that helps the user find the information in the policy. 
Use wording such as:

   “If you’d like to read the full details, you can refer to the policy section(s) mentioned above.”
Or:
   “You can also review the full definitions for more context in the pages noted earlier.”

This reference must always come AFTER your main explanation, never before it.
It should reinforce the summary, not replace it.
Do NOT only say ‘refer to section X’ — always place it after providing meaningful information.

9. Brand / Company Name Suppression:
    - If the policy text contains any insurer or company name such as “Income Insurance Limited”, 
      do NOT repeat or mention the company name in your response.
    - You may replace such names with phrases like:
        “the insurer”, “the policy provider”, or “the insurance company”.
    - Never quote or state the company’s legal name anywhere in the answer.
    - This rule applies even if the user asks directly about the company name.
    
    
Tone:
- Warm, calm, and professional.
- Sound like a knowledgeable human support agent.
- Never lecture or scold the user.
- Keep answers concise but informative.
"""


def build_user_prompt(_question, _context):
    return f"""
You will answer a question using ONLY the information in the CONTEXT below.

If the answer is available, provide a helpful, friendly explanation that summarises the relevant details.
If multiple relevant sections appear, combine them.

If you reference any section, page number, or subsection:
- Always provide a meaningful summary of that section based on the context.
- Never respond with only “refer to section X”.

If the answer is not directly found:
- Provide the closest related information you can find in the context.
- Politely inform the user that the specific detail is not mentioned.
- Suggest contacting customer service for full clarification.

### USER QUESTION
{_question}

### POLICY CONTEXT
{_context}

"""


def answer_question(_question, _policy_id=None, _top_k=5, _pre_retrieved_chunks=None):
    """ Answer user question using retrieved chunks from vectordb and LLM """
    if _pre_retrieved_chunks is None:
        retrieved = search_chunks(_question, _top_k=_top_k, _policy_id=_policy_id)
    else:
        retrieved = _pre_retrieved_chunks

    # build context
    context_blocks = []
    for each_retrieve in retrieved:
        meta = each_retrieve["metadata"]
        context_blocks.append(
            f"### SOURCE CHUNK\n"
            f"Policy: {meta.get('policy_id')}\n"
            f"Section: {meta.get('section')}\n"
            f"Subsection: {meta.get('subsection')}\n"
            f"Pages: {meta.get('page_start')}-{meta.get('page_end')}\n"
            f"FusionScore: {each_retrieve.get('fusion_score', 0):.2f}\n\n"
            f"{each_retrieve['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    # build user prompt for the LLM
    prompt = build_user_prompt(_question, context)

    # LLM call
    client = openai_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": retrieved
    }

