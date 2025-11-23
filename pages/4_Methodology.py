import streamlit as st

st.set_page_config(layout="wide", page_title="Methodology")

st.title("📘 Methodology")
st.markdown("""
A detailed explanation of how this Insurance Policy AI Assistant works, including its RAG pipeline, preprocessing steps, retrieval logic, reranking, and LLM prompting.

> :orange[Note that this is an individual-based capstone assignment and the domain chosen is about Insurance policies.]

""")
st.info("""
The project has also been a journey of discovery - experimenting with and trying to understand concepts behind building RAG applications. 

It has involved quite some exploratory work: investigating different approaches, studying course materials, iterating with ChatGPT, and researching techniques and solutions across various resources.
""")
st.markdown("---")

# overview
st.header("🔎 Overview")

st.markdown("""
Insurance policies are long, technical, and complex documents.  
This application uses a :blue[Retrieval-Augmented Generation (RAG)] pipeline to allow users to ask natural-language questions and receive answers derived from the preloaded official policy documents.

### Key Goals
- ✔ Improve policy understanding.  
- ✔ Reduce misinformation.  
- ✔ Provide transparent, document-grounded answers.  
- ✔ Keep responses friendly and easy to read.  
- ✔ Reduce hallucination.  
""")

st.subheader("📌 High-Level Flowchart")

# mermaid diagram
rag_pipeline = """
flowchart LR
    A[User Question] --> B[Query Normalization]
    B --> C[Dense Retrieval/Embedding]
    C --> D[Vector Search ChromaDB]
    B --> E[BM25 Sparse Retrieval]
    D --> F{Hybrid - Dense + Sparse}
    E --> F
    F --> G[Lightweight Reranker]
    G --> H[Top-K Selected Chunks]
    H --> I[Prompt Builder]
    I --> J[LLM Response Generation]
    J --> K[Final Answer - Streamlit]
"""

st.image("pages/images/rag.svg", caption="RAG Pipeline Overview", width="stretch")

st.markdown("---")

st.header("📃 PDF Ingestion & Chunking")
st.markdown("""
Policy documents come in a lot of different formats, layout, and complexity. 

For this project, some simpler sample policies were chosen, and they are relatively well-structured, with clear sections, subsections, and consistent formatting.

To extract meaningful content, the system performs multi-stage preprocessing:

1. 🔹 Extract PDF page content
    - Extract text page-by-page from the PDF.

2. 🔹 Line Flattening
    - Convert each page into structured lines:
    ```
    { "page": X, "line": "text..." }
    ```

3. 🔹 Group by Sections
    - Detect main big section using regex patterns.
""")
st.image("pages/images/chunking1.png")
st.markdown("""

4. 🔹 Group by Subsections
    - Within each section, detect subsections using regex patterns.
""")
st.image("pages/images/chunking2.png")
st.markdown("""
5. 🔹 Convert to Paragraph
    - Remove empty lines and new lines to form continuous paragraphs.

6. 🔹 Hybrid Chunking
    - LangChain RecursiveCharacterTextSplitter (for the section and subsection chunks) 
        - chunk_size = 600
        - overlap = 120
    - Further merge specific chunks(by keywords) into one big chunk. (:violet[Refer to the [Learnings](#learnings) section for the rationale behind this step.])
        - The last sections of the policy often contain big tables of information, multi-column text and long :blue[Definitions] of certain terms that span many lines.
        - Naive chunking breaks these important long definitions into smaller pieces, losing context.
        - Define specific keywords to identify such chunks and merge them into one big chunk.
        - Example:
          ```
          TITLE_KEYWORDS: ["definition", "definitions", "define", "meaning", "glossary"......]
          CONTENT_KEYWORDS: ["diagnosed", "diagnosis", "criteria", "confirmed by a specialist"......]
          MEDICAL_HINT_KEYWORDS: ["carcinoma", "metastasis", "tumour", "tumor", "malignant"......]
          ```
7. 🔹 Chunk object
    ```python
    class Chunk:
        policy_id: str
        section: str
        subsection: str
        page_start: int
        page_end: int
        chunk_index: int
        text: str
    ```
""")
st.markdown("---")

st.header("📦 Vector Database (ChromaDB)")
st.markdown("""

#### :blue[Why use ChromaDB?]

- Persistent on disk
- Fast retrieval
- Lightweight for Streamlit Cloud
- Supports metadata filtering

#### :blue[Stored Metadata Per Chunk]
- policy id
- section title
- subsection title
- page start/end
- chunk index
- text

These metadata fields are used later for context reconstruction and referencing.

#### :blue[Project]
- All the 5 sample insurance policies are already preprocessed and stored in ChromaDB as the Vector DB and pushed to the Git Repository.
- On application startup, the existing ChromaDB collection is loaded for retrieval.
- There is no rebuilding of the Vector DB at runtime.
- The code to build the Vector DB from scratch is available in the :blue[helper_functions/vectordb.py(create_vector_db function)] file for reference.
""")

st.markdown("---")

st.header("🔍 Retrieval")

st.markdown("""
The retrieval process combines both Dense Retrieval (using embeddings) and BM25 lexical scoring to improve the relevance of retrieved chunks.

Attempt to use :blue[Hybrid Retrieval] for better results:

1. 🔹 Dense Retrieval (Embeddings)
    - We embed the user query using OpenAI :blue[text-embedding-3-small] model.
    - Then query the Vector DB for top-K semantically similar chunks.

2. 🔹 BM25 lexical scoring (:violet[Refer to the [Learnings](#learnings) section for the rationale behind this step.])
    - After retrieving the results from the Dense Retrieval step, we apply BM25 keyword scoring on the candidate chunks.
     - BM25 enhances keyword recall, especially for:
        - medical terms
        - acronyms
        - definitions
        - multi-line structured text

3. 🔹 Fusion of Results
    - Combine BM25 + embedding scores:
        - fusion_score = 0.6 * dense + 0.4 * bm25
    - Then sort and select the best chunks.

4. 🔹 Lightweight hybrid reranker using BM25 + dense similarity
    - Local reranking layer combines multiple relevance signals to choose the best chunks before sending them to the LLM.
        - keyword relevance
        - semantic relevance match
        - chunk structure 
""")
st.markdown("---")

st.header("✅ Response Generation & Prompt Engineering")

st.markdown("""
The final step uses the OpenAI LLM to generate a friendly and accurate response.

No fancy or advanced prompting techniques are being used since answers should be concise and directly from the policy text.

#### :blue[System Prompt]

- No hallucination (or at least try to minimize it)
- The assistant must answer only from the retrieved policy chunks.
- The assistant must say so if the policy does not contain the answer.
- Friendly and professional tone
- Responses follow a warm, informative, customer-service style.
- Prohibits referencing the mentioned insurance company name.
- Each answer should:
    - give a clear summary
    - elaborate based on the retrieved chunks
    - avoid too much jargon
    - optionally point the user to the sections/pages where detail appears
- Code Snippet:
  ```python
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
    .
    . <refer to actual code for more details>
    .
    Tone:
    - Warm, calm, and professional.
    - Sound like a knowledgeable human support agent.
    - Never lecture or scold the user.
    - Keep answers concise but informative.
  ```
  
  
#### :blue[User Prompt Builder]
- Embeds the selected chunks
- Constructs a clear prompt with context + user question
- Natural-Language Rewriting Techniques
- Follow-up question support from Streamlit chat
    
    | Layer               | Conversation history        | Description                                                  |
    |---------------------|-----------------------------|--------------------------------------------------------------|
    | Retrieval           | Yes (only the previous one) | Helps the model find correct context for follow-up questions |
    | Response generation | No                          | Prevents contamination or repeated text                      |

- Code:
  ```python
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
  ```
""")
st.markdown("---")

st.header("🧠 Conversation Memory")
st.markdown("""
The application tries to support short contextual follow-ups, but avoid long conversation history because insurance answers must remain strictly policy-grounded.

#### :blue[Strategy]
- Remember only the user’s previous question (if relevant)
- Do not append long session history to retrieval
- Conversation resets automatically when switching policies 

This keeps answers accurate and prevents cross-policy contamination.
""")
st.markdown("---")

st.header("▶️ Streamlit UI")

st.markdown("""

#### :blue[Components]

- Secure login page (username/password via st.secrets)
- Sidebar:
  - Policy selector
  - top_k selector
  - Clear chat
  - Logout
- Main Chat Interface
- Pages:
  - View All Policies
  - About Us
  - Methodology (this page)

#### :blue[PDF Viewer]

- Inbuilt PDF preview using :blue[streamlit_pdf_viewer] library.

#### :blue[State Handling]
- session_state.messages
- session_state.last_user_question
- session_state.authenticated
- Chat reset when policy changes

""")
st.image("pages/images/chat.png")
st.markdown("---")

st.header("🔀 Process Flowchart")
st.subheader("🧠 Chat With AI Assistant")

flow1 = """
flowchart LR
    A[User enters question] --> B[Normalize Question]
    B --> C[Embed Question]
    C --> D[Vector Search]
    B --> E[BM25 Keyword Search]
    D --> F[Fusion Ranking]
    E --> F
    F --> G[Rerank Results]
    G --> H[Top Chunks]
    H --> I[LLM Answer Prompt]
    I --> J[Generate Final Answer]
    J --> K[Show in Chat UI]
"""
st.image("pages/images/flow1.svg", caption="Chat Flowchart", width="stretch")
st.markdown("---")

st.header("❔Sample Questions for Chat Assistant")
st.markdown("""

#### :blue[Medical/Critical Illness] 
(policies: cancer_care or lady_360 or silver_secure)

- What are the diagnostic criteria for major cancer?
- What does carcinoma-in-situ mean in this policy?
- What are the conditions for early stage breast cancer?
- How does the policy classify advanced stage cancer?
- What laboratory tests are required for diagnosis?
- How does the policy define a malignant tumour?
- What cancers are excluded from coverage?
- What are the diagnostic tests required for heart attack?
- How does the policy define Alzheimer’s disease?
- When does the policy terminate?
- Can the insurer refuse renewal?
- What is the payout for total and permanent disability?

#### :blue[Private Car Insurance] 
(policy: private_car)

- What does the policy cover for vehicle damage?
- What are the limits for damage to third-party property?
- Are accessories covered? Under what conditions?
- Does the policy cover windscreen damage?
- Does the policy cover unnamed drivers?
- What is the meaning of named driver?
- How is market value determined during claims?”

#### :blue[Female-Specific Policy Questions] 
(policy: lady_360)

- What conditions are covered under the female illness benefit?
- Is breast cancer covered? Under what stage?
- What maternity-related complications are covered?
- What are the exclusions for female-specific illnesses?

""")
st.markdown("---")

st.header("Learnings")
st.markdown("""
#### 🧪 :violet[Tests]
In the initial phase, the chunking strategy involve splitting the PDF text into chunks by big sections, followed by further splitting into smaller subsection chunks using LangChain's RecursiveCharacterTextSplitter.

After taking a random policy for testing, it was found that this approach led to suboptimal results, especially for sections containing long definitions or structured data.

1. One of the document section - :green[Section 1: What your policy covers]
    ```
    1 What your policy covers
    a. Dread disease benefit

    Dread disease consists of early stage, intermediate stage and advanced stage dread disease.
    If the insured is diagnosed with a dread disease by a specialist during the term of this policy, we will pay the benefit according to its severity level shown in Table 1 if you have chosen Protect 100.......
    ```
2. In another big section - :green[6. Definition of early, intermediate and advanced stage dread diseases]
    ```
    6 Definition of early, intermediate and advanced stage dread diseases
    .
    <followed by big tables and a lot of information>
    ```
3. Prompt: :green[what is the definition of early, intermediate and advanced stage dread diseases?]
   - The LLM response given:
   ```
   ANSWER: 
   The policy text provided does not include the specific definitions of early stage, intermediate stage, and advanced stage dread diseases. 
   However, it does state that dread disease consists of these three stages and outlines the benefits associated with them in the context of the policy (section 1, subsection a). 
   If you need more detailed definitions or specific conditions for each stage, please refer to the full policy document or consult with your insurance provider.
   ```
4. From the above, the response seems to have missed the entire Section 6 which contains the detailed definitions.

#### 🔧 :violet[Refining part 1: BM25 keyword scoring]

I was introduced to this concept about combining dense and some sort of sparse retrieval by ChatGPT.

Dense retrieval favours similarity rather than semantic relevance.

Upon investigation, we could try to improve keyword matching for such queries, thus :green[BM25 lexical scoring] was added as a complementary retrieval/reranking? method alongside dense retrieval.

1. BM25 is a keyword-based ranking algorithm that is not typically a reranker itself but is widely used as the initial retrieval step in a multi-stage hybrid search and reranking pipeline.
2. Use of :green[rank_bm25] python library to compute BM25 scores for candidate chunks retrieved from dense retrieval.
3. As far as I understood it:
    - BM25 score (sparse relevance)
    - Dense embedding similarity
    - Keyword scoring
    - Fusion scoring

4. After implementing BM25 scoring on top of dense retrieval, the same prompt now yields:
    ```
    ANSWER: 
    The definitions of early, intermediate, and advanced stage dread diseases are provided in Section 6 of the policy. 
    Here is a summary of the definitions: 
    ### Major Cancer 
    - **Early Stage**: 
        - Carcinoma-in-situ (CIS) is defined as the focal autonomous new growth of carcinomatous cells that is confined to the cells in which it originated. 
        It has not yet resulted in the invasion and/or destruction of surrounding tissues. 
        - ‘Invasion’ refers to the infiltration and/or active destruction of normal tissue beyond the basement membrane. 
    - **Intermediate Stage**: 
        - The policy text does not provide a specific definition for intermediate stage dread diseases in the retrieved context. 
    - **Advanced Stage**: 
        - The policy text does not provide a specific definition for advanced stage dread diseases in the retrieved context. 
    For more detailed definitions, please refer to Section 6 of the policy, as it may contain additional information
    ```
5. We can now see that the response has improved and the top_k chunks regarding :green[6. Definition of early, intermediate and advanced stage dread diseases] are being returned as the response.

6. However, the information about intermediate and advanced stages still seems missing.

7. After some investigations and showing output to ChatGPT, the reason found was that the initial chunking strategy has split the long Definition section blocks into smaller chunks, and the important context is lost.
   
   And thus the LLM apparently sees this as incomplete information.
```
Top retrieved chunks: 
1. section=4 What you need to be aware of | subsection=b Dread disease benefit | pages=11-12 | distance=0.3025 | dense=0.95 | bm25=1.00 | fusion=0.97 4 What you need to be aware of b Dread disease benefit We only cover the dread disease we define in this policy. The full definition of an early stage, intermediate stage or advanced stage dread disease covered and the circumstances in which you can
2. section=1 What your policy covers | subsection=a Dread disease benefit | pages=1-4 | distance=0.2879 | dense=1.00 | bm25=0.85 | fusion=0.94 1 What your policy covers a Dread disease benefit Dread disease consists of early stage, intermediate stage and advanced stage dread disease. If the insured is diagnosed with a dread disease by a specialist during the term of this policy, we will pa
3. section=1 What your policy covers | subsection=a Dread disease benefit | pages=1-4 | distance=0.3059 | dense=0.93 | bm25=0.63 | fusion=0.81 We will pay the early and/or intermediate stage dread disease under this benefit, subject to the following: • dread disease benefit has not ceased at the time of any payment of the benefit; • the insured survives at least 7 days after the date of di
4. section=6 Definition of early, intermediate and advanced stage dread diseases | subsection=main | pages=17-37 | distance=0.4031 | dense=0.61 | bm25=0.94 | fusion=0.74 6 Definition of early, intermediate and advanced stage dread diseases main 6.1 Major Cancer Early Stage Intermediate Stage Advanced Stage • Carcinoma-in-situ (CIS) Carcinoma-in-situ (CIS) means the focal autonomous new growth of carcinomatous cells
5. section=1 What your policy covers | subsection=a Dread disease benefit | pages=1-4 | distance=0.3356 | dense=0.83 | bm25=0.54 | fusion=0.71 We will pay the advanced stage dread disease under this benefit, subject to the following: • dread disease benefit has not ceased at the time of any payment of the benefit; • the insured survives at least 7 days after the date of diagnosis or date of...
```

#### 🔧 :violet[Refining part 2: Merging big Definition chunks]

Looking at the sample Insurance PDFs, they often contain definition-heavy sections such as:
- Definition of an early, intermediate and advanced stage of major cancer
- Definition of cancer treatment
- Definition of insured events
- Definition of early, intermediate and advanced stage dread diseases
- Some just have a big section simply titled "Definitions"

These sections are dense, long, and sometimes full of bullet lists and tables, which makes them difficult for RAG systems to chunk correctly or susceptible to partial retrieval of context.

The solution is to try to identify such definition-heavy chunks after the initial section and subsection grouping, and merge them into one big chunk to preserve context.

1. Use of specific keywords to identify definition-like chunks. (on best effort and only pertaining to the style of such policies)
2. Keywords code:
    ```python
    DEF_TITLE_KEYWORDS = [
        "definition", "definitions", "define", "meaning", "glossary",
        "diagnostic criteria", "staging", "stage", "classification"
    ]

    DEF_CONTENT_KEYWORDS = [
        "means", "refers to", "shall mean", "is defined as",
        "diagnosed", "diagnosis", "criteria", "confirmed by a specialist",
        "must meet all of the following"
    ]

    MEDICAL_HINT_KEYWORDS = [
        "carcinoma", "metastasis", "tumour", "tumor", "malignant",
        "chemotherapy", "radiotherapy", "mri", "ct scan", "biopsy",
        "coronary", "angiography", "renal", "neurological",
        "parkinson", "alzheimer", "copd", "stroke"
    ]
    ```
3. If a chunk contains 
    - a definition title OR 
    - contains definition content OR 
    - contains medical indicators THEN
    - merge this chunk with the surrounding chunks until the definition block ends.

4. After implementing this merging step, the same prompt now yields:
    ```
    ANSWER:
    The definitions of **early stage**, **intermediate stage**, and **advanced stage** dread diseases are outlined in the policy and refer to the severity of the condition as diagnosed by a specialist. Here’s a summary of each stage:

    ### Early Stage Dread Disease
    - **Definition**: This stage typically involves a diagnosis of a dread disease that is not yet severe enough to cause significant impairment or require extensive treatment.
    - **Example**: A diagnosis of early Alzheimer’s disease or a large asymptomatic aortic aneurysm.
    
    ### Intermediate Stage Dread Disease
    - **Definition**: This stage indicates a progression of the disease where symptoms may be more pronounced, and the condition requires more intensive management.
    - **Example**: Moderately severe Alzheimer’s disease or a large asymptomatic abdominal aortic aneurysm confirmed by imaging.
    
    ### Advanced Stage Dread Disease
    - **Definition**: This stage represents the most severe form of the disease, often leading to significant health complications and requiring urgent medical intervention.
    - **Example**: Advanced Alzheimer’s disease or end-stage liver failure.
    
    ### Key Points
    - The policy specifies that benefits are paid according to the severity level of the dread disease diagnosed.
    - The conditions for claiming benefits vary based on the stage of the disease, including survival periods and the number of claims allowed.
    
    If you’d like to read the full details, you can refer to the definitions section of the policy, which provides comprehensive information on each stage and the specific conditions covered.
    ```
5. The response at least now contains information about all three stages of the diseases.
6. Though I believe that the RAG and prompting could be further fined-tuned, the response now is an improvement from the initial one.

#### 🔧 :violet[Take-aways]

Long story short: 😊
- Hybrid retrieval (dense + key scoring) can help improve relevance for certain queries.
- Long definition-heavy sections may need special handling to preserve context.
- Learnt quite a few things about RAG pipelines, retrieval strategies, and prompt engineering!
- Also excluded this sample policy from the project as this particular structure seems complicated for now.

""")

st.success("End of Methodology")