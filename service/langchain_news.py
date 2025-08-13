# service/langchain_news.py

from __future__ import annotations
import re
import os
import ast
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

# --- LangChain imports (prefer langchain_openai; fallback for older setups) ---
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover
    # Older LangChain versions
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.embeddings import OpenAIEmbeddings  # type: ignore

from langchain.schema import Document
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore



# FAISS native
import faiss


# =========================
# API & Model Initialization
# =========================

def load_api_key(path: Optional[str] = None) -> str:
    """
    Load the OpenAI API key from:
    1) path (if provided),
    2) env var OPENAI_API_KEY,
    otherwise raise.
    """
    if path:
        with open(path, "r") as f:
            key = f.read().strip()
            if not key:
                raise ValueError("API key file is empty.")
            return key

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "OpenAI API key not found. Set OPENAI_API_KEY or provide api_key_path."
        )
    return key


# def init_models(
#     api_key_path: Optional[str] = None,
#     chat_model: str = "gpt-4o-2024-05-13",
#     embedding_model: str = "text-embedding-3-large",
# ):
#     """
#     Return (llm, embedder) configured with your key.
#     """
#     api_key = load_api_key(api_key_path)

#     llm = ChatOpenAI(model=chat_model, openai_api_key=api_key)
#     embedder = OpenAIEmbeddings(model=embedding_model, openai_api_key=api_key)
#     return llm, embedder

def init_models(
    api_key_path: Optional[str] = None,
    chat_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """
    Return (llm, embedder) configured with your key.
    Either parameter can be None if not needed.
    """
    api_key = load_api_key(api_key_path)

    llm = None
    embedder = None

    if chat_model:
        llm = ChatOpenAI(model=chat_model, openai_api_key=api_key)

    if embedding_model:
        embedder = OpenAIEmbeddings(model=embedding_model, openai_api_key=api_key)

    return llm, embedder

# =========================
# Data Loading
# =========================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load DataFrame from .csv, .xlsx, or .parquet.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df


# =========================
# Helpers
# =========================
def _ensure_vector(x: Any) -> np.ndarray:
    """
    Ensure a single-row embedding is a 1D float32 numpy array.
    Accepts list, np.ndarray, or stringified list.
    """
    if isinstance(x, str):
        # try to parse stringified list
        x = ast.literal_eval(x)
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Embedding must be 1D per row.")
    return arr


def _ensure_matrix(series: pd.Series) -> np.ndarray:
    """
    Convert a Series of embeddings into a 2D float32 matrix.
    """
    vecs = [_ensure_vector(e) for e in series]
    mat = np.vstack(vecs).astype(np.float32)
    return mat


def _documents_from_df(df: pd.DataFrame) -> List[Document]:
    """
    Build LangChain Documents from the required columns.
    """
    docs: List[Document] = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=row["Descriptions"],
                metadata={
                    "File_name": row.get("File_name", ""),
                    "Description": row.get("Descriptions", ""),
                    "Code": row.get("Code", ""),
                    "Dependencies": (row.get("Dependencies", "") or "").strip(),
                    "Functions": row.get("Functions", ""),
                    "Docstring_summary": row.get("Docstring_summary", ""),
                    "Notes": row.get("Notes", ""),
                },
            )
        )
    return docs

def first_sentence(text: str) -> str:
    """
    Return the first sentence from text.
    Handles '.', '?', '!', and CJK '。！？'.
    """
    if not text:
        return ""
    t = " ".join(text.split()).strip()
    # split on ASCII or CJK sentence enders followed by whitespace or end
    parts = re.split(r'(?<=[\.\?\!。！？])\s+', t, maxsplit=1)
    return parts[0] if parts else t


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

ALLOWED_DESIGNS = {"fixed design", "group sequential design", "not sure"}

def _normalize_design_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "group" in s or "sequential" in s or "gs" in s:
        return "group sequential design"
    if "fixed" in s:
        return "fixed design"
    if "not sure" in s or "unsure" in s or "uncertain" in s:
        return "not sure"
    if s in ALLOWED_DESIGNS:
        return s
    return "not sure"

ALLOWED_INTENTS = {"code modification", "conceptually explanation", "not sure"}

def _normalize_intent_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    # tolerate tiny wording variances
    if "code" in s or "modify" in s or "patch" in s or "edit" in s:
        return "code modification"
    if "concept" in s or "explain" in s or "explanation" in s or "theory" in s:
        return "conceptually explanation"
    if "not sure" in s or "unsure" in s or "uncertain" in s:
        return "not sure"
    # last resort: try exact allowed
    if s in ALLOWED_INTENTS:
        return s
    return "not sure"

def _parse_dependencies(dep_str: str) -> List[str]:
    """
    Parse dependencies from free-form text like:
      "check_arg.R, gs_b.R, gs_design_combo.R"
    Returns clean file tokens.
    """
    if not dep_str:
        return []
    s = str(dep_str).strip()
    if s.lower() in {"none", "null", "na", "n/a"}:
        return []
    parts = [re.sub(r"^[\"'\s]+|[\"'\s]+$", "", p) for p in re.split(r"[,\n;]+", s)]
    return [p for p in parts if p]


def _dep_basename(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0].lower()

# =========================
# Vector Stores (FAISS)
# =========================

def build_main_store_from_df(df: pd.DataFrame) -> FAISS:
    """
    Build a FAISS vector store backed by precomputed embeddings
    contained in df['Embedding'].
    Note: We use a dummy embedding function because we never embed here.
    """
    if "Embedding" not in df.columns:
        raise KeyError("DataFrame must contain an 'Embedding' column.")

    # Documents
    documents = _documents_from_df(df)

    # Embedding matrix
    embedding_matrix = _ensure_matrix(df["Embedding"])
    dim = embedding_matrix.shape[1]

    # FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)

    # DocStore
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # Dummy embedding function: we never call .embed_documents on this store.
    def _dummy_embed(_: List[str]) -> List[List[float]]:
        raise NotImplementedError("This store relies on precomputed embeddings only.")

    vector_store = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=_dummy_embed,
    )
    return vector_store

def build_stores_from_labeled_df(df: pd.DataFrame) -> Dict[str, FAISS]:
    """
    Build three FAISS vector stores keyed as:
      - "all"   : labels {0,1,2}
      - "fixed" : labels {0,2}
      - "gs"    : labels {1,2}
    Falls back to "all" only if 'label' column is missing.
    """
    if "label" not in df.columns:
        store_all = build_main_store_from_df(df)
        return {"all": store_all, "fixed": store_all, "gs": store_all}

    def _subset(df_, labels):
        return df_[df_["label(0:fix sample, 1:group sequential, 2:neutral )"].isin(labels)].reset_index(drop=True)

    all_df   = _subset(df, [0,1,2])
    fixed_df = _subset(df, [0,2])
    gs_df    = _subset(df, [1,2])

    stores = {
        "all":   build_main_store_from_df(all_df if len(all_df)   else df),
        "fixed": build_main_store_from_df(fixed_df if len(fixed_df) else all_df if len(all_df) else df),
        "gs":    build_main_store_from_df(gs_df if len(gs_df)    else all_df if len(all_df) else df),
    }
    return stores

# ---------- PDF (temporary) ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF via PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise ImportError(
            "PyMuPDF (fitz) is not available. Install a compatible wheel to enable PDF extraction."
        ) from e
    with fitz.open(pdf_path) as doc:  # type: ignore
        pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def extract_text_from_txt(txt_path: str, encodings: list[str] | None = None) -> str:
    """Read plain text file with fallback encodings."""
    if encodings is None:
        encodings = ["utf-8", "utf-16", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            with open(txt_path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_err = e
    raise ValueError(
        f"Failed to read text file {txt_path} with tried encodings. Last error: {last_err}"
    )

def read_upload_text(file_path: Optional[str] = None,
                     pdf_path: Optional[str] = None,
                     max_chars: int = 60_000) -> Optional[str]:
    """
    Return plain text of the uploaded file (PDF/TXT) for prompt injection.
    Truncates to max_chars to avoid blowing the context window.
    """
    _path = file_path or pdf_path
    if not _path:
        return None

    ext = os.path.splitext(_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(_path)
    elif ext == ".txt":
        text = extract_text_from_txt(_path)
    else:
        return None  # unsupported types are ignored

    text = text or ""
    if not text.strip():
        return None

    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n...[TRUNCATED]..."
    return text


def build_temp_faiss_for_text(
    text: str,
    file_name: str,
    embedder: OpenAIEmbeddings,
    chunk_size: int = 1500,
    chunk_stride: Optional[int] = None,
) -> FAISS:
    """Build a temporary FAISS store for an arbitrary text blob (PDF/TXT already extracted)."""
    if not (text or "").strip():
        raise ValueError("Provided text is empty.")
    cs = chunk_size
    step = chunk_stride or cs
    chunks: List[str] = [text[i : i + cs] for i in range(0, len(text), step)]

    documents: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_id": f"{file_name}_chunk_{i}",
                "File_name": file_name,
                "Description": f"Extracted from uploaded file: {file_name}",
                "Code": "",
                "importance": 1,
                "source": "upload",     # <--- add
                "is_upload": True, 
            },
        )
        for i, chunk in enumerate(chunks, start=1)
    ]

    vectors = embedder.embed_documents([d.page_content for d in documents])
    import numpy as np  # 若上面已全局 import，则可省略
    vectors = np.asarray(vectors, dtype=np.float32)
    dim = vectors.shape[1]

    import faiss  # 若上面已全局 import，则可省略
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    from langchain.docstore import InMemoryDocstore  # 若上面已全局 import，则可省略
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    def _dummy_embed(_: List[str]) -> List[List[float]]:
        raise NotImplementedError("Temporary store uses precomputed embeddings for chunks.")

    return FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=_dummy_embed,
    )


def build_temp_faiss_for_file(
    file_path: str,
    embedder: OpenAIEmbeddings,
    chunk_size: int = 1500,
    chunk_stride: Optional[int] = None,
) -> FAISS:
    """Dispatch by extension: .pdf -> extract_text_from_pdf ; .txt -> extract_text_from_txt"""
    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported upload type: {ext}. Only .pdf and .txt are supported.")
    return build_temp_faiss_for_text(
        text=text,
        file_name=file_name,
        embedder=embedder,
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
    )

# =========================
# RAG Orchestrator
# =========================
class RAGHelper:
    """
    Encapsulates your main FAISS store (with precomputed embeddings),
    a query embedding model, and an LLM.

    Supports optional, per-call PDF augmentation.
    """

    def __init__(self, 
                 stores: Dict[str, FAISS],
                 llm: ChatOpenAI,
                 embedder: OpenAIEmbeddings,
                 raw_df: Optional[pd.DataFrame] = None,
                 file_col: str = "File_name",
                 code_col: str = "Code",
                 llm_intent=None):
        
 
        self.stores = stores
        self.llm = llm
        self.embedder = embedder
        try:
            self.router_llm = ChatOpenAI(model=llm.model_name, openai_api_key=llm.openai_api_key, temperature=0)
        except Exception:
            self.router_llm = llm

        self.llm_intent = llm_intent or llm 
        self.raw_df = raw_df
        self.file_col = file_col
        self.code_col = code_col

        # build doc indices (existing)
        self.filename_index = {}
        self.basename_index = {}
        try:
            all_store = self.stores.get("all") or next(iter(self.stores.values()))
            backing = getattr(all_store.docstore, "_dict", {})
            for _, doc in backing.items():
                m = doc.metadata or {}
                if m.get("is_upload") or (m.get("source") == "upload"):
                    continue
                fname = (m.get("File_name") or "").strip()
                if fname:
                    self.filename_index.setdefault(fname, doc)
                    base = os.path.splitext(os.path.basename(fname))[0].lower()
                    self.basename_index.setdefault(base, doc)
        except Exception:
            pass

        # build raw-df indices (new)
        self.df_file_index = {}
        self.df_base_index = {}
        if self.raw_df is not None and self.file_col in self.raw_df.columns:
            for _, row in self.raw_df.iterrows():
                fname = str(row.get(self.file_col, "")).strip()
                if not fname:
                    continue
                self.df_file_index.setdefault(fname, row)
                base = os.path.splitext(os.path.basename(fname))[0].lower()
                self.df_base_index.setdefault(base, row)

    # --------------------
    # Internal utilities
    # --------------------
    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.embed_query(query)
        return np.asarray(vec, dtype=np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int = 2,                # limit ONLY the internal codebase hits
        pdf_path: Optional[str] = None,
        file_path: Optional[str] = None,
        design_key: str = "all",   # "fixed" | "gs" | "all"
        ) -> List[Document]:
        """
        Return top_k from the main store, plus ALL matching chunks from an uploaded file.
        """
        # 1) embed query
        qvec = self._embed_query(query)

        # 2) main store (internal codebase): clamp to top_k
        main_store = self.stores.get(design_key, self.stores.get("all"))
        main_hits: List[Document] = main_store.similarity_search_by_vector(qvec, k=top_k)


        # 3) uploaded file (pdf/txt): return ALL chunks (no cap)
        uploaded_hits: List[Document] = []
        _path = file_path or pdf_path
        if _path:
            tmp_store = build_temp_faiss_for_file(_path, self.embedder)

            # figure out how many chunks there are, then ask for all of them
            try:
                # works for FAISS wrapper we built: "0..N-1" ids
                n_chunks = len(tmp_store.index_to_docstore_id)
            except Exception:
                # fallback to docstore size
                n_chunks = len(getattr(tmp_store.docstore, "_dict", {})) or 50  # sane fallback

            # request ALL chunks (you can also set a high ceiling like 1000)
            uploaded_hits = tmp_store.similarity_search_by_vector(qvec, k=n_chunks)

        # 4) merge (keep internal first, then uploaded)
        combined = main_hits + uploaded_hits

        # 5) dedupe by page_content (optional; keeps unique text blocks)
        seen = set()
        deduped: List[Document] = []
        for d in combined:
            pc = d.page_content
            if pc in seen:
                continue
            seen.add(pc)
            deduped.append(d)

        # 6) DO NOT slice to top_k; we want all uploaded chunks + top_k main hits
        return deduped

    def _resolve_dep_doc_or_row(self, dep_name: str):
        """Return (mode, obj) where mode in {'doc','row',None}."""
        # 1) exact doc
        d = self.filename_index.get(dep_name)
        if d:
            return "doc", d
        # 2) basename doc
        d = self.basename_index.get(_dep_basename(dep_name))
        if d:
            return "doc", d
        # 3) exact df row
        if self.df_file_index:
            r = self.df_file_index.get(dep_name)
            if r is not None:
                return "row", r
        # 4) basename df row
        if self.df_base_index:
            r = self.df_base_index.get(_dep_basename(dep_name))
            if r is not None:
                return "row", r
    
    def _block_from_doc(self, doc: Document, tag: str = "Dependency") -> str:
        m = doc.metadata or {}
        fname = (m.get("File_name") or "").strip()
        desc  = m.get("Description", "")
        code  = m.get("Code", "")
        funcs = m.get("Functions", "")
        return f"[{tag}: {fname}]\nDescription: {desc}\n\n{code}\nAvailable functions: {funcs}"

    def _block_from_row(self, row: pd.Series, tag: str = "Dependency") -> str:
        fname = str(row.get(self.file_col, "")).strip()
        code  = str(row.get(self.code_col, "") or "")
        desc  = str(row.get("Descriptions", row.get("code detailed Descriptions", row.get("Description",""))))
        funcs = str(row.get("Functions", row.get("Funtctions", row.get("Funtctions included",""))))
        return f"[{tag}: {fname}]\nDescription: {desc}\n\n{code}\nAvailable functions: {funcs}"


    def build_context(
        self,
        docs: List[Document],
        include_upload: bool = False,
        max_deps_per_main: int = 10,
        transitive: bool = False,
        max_code_chars_per_file: int = 12000,   # optional budget
        ) -> str:
        blocks: List[str] = []
        seen_files: set[str] = set()

        def _trim_code(block: str) -> str:
            if not max_code_chars_per_file or len(block) <= max_code_chars_per_file:
                return block
            return block[:max_code_chars_per_file] + "\n...[TRUNCATED]..."

        for d in docs:
            meta = d.metadata or {}
            if not include_upload and (meta.get("is_upload") or meta.get("source") == "upload"):
                continue

            fname = (meta.get("File_name") or "").strip()
            if fname and fname in seen_files:
                continue
            seen_files.add(fname)

            # main file block (doc)
            main_block = self._block_from_doc(d, tag="File")
            blocks.append(_trim_code(main_block))

            # deps
            dep_str = meta.get("Dependencies", "") or ""
            if dep_str and not dep_str.strip().lower() in {"none","null","na","n/a"}:
                print(f"[CTX] {fname} deps raw -> {dep_str}")
            deps = _parse_dependencies(dep_str)
            if not deps:
                continue

            added = 0
            queue = list(deps)
            while queue and added < max_deps_per_main:
                dep_name = queue.pop(0)
                mode, obj = self._resolve_dep_doc_or_row(dep_name)
                print(f"[CTX] resolve '{dep_name}' ->", "DOC" if mode=="doc" else "ROW" if mode=="row" else "MISS")
                if not obj:
                    continue

                if mode == "doc":
                    dep_meta = obj.metadata or {}
                    dep_fname = (dep_meta.get("File_name") or "").strip()
                else:  # row
                    dep_fname = str(obj.get(self.file_col, "")).strip()

                if dep_fname in seen_files:
                    continue
                seen_files.add(dep_fname)

                block = self._block_from_doc(obj, tag="Dependency") if mode == "doc" else self._block_from_row(obj, tag="Dependency")
                blocks.append(_trim_code(block))
                added += 1

                if transitive:
                    # enqueue transitive deps from whichever representation we used
                    if mode == "doc":
                        sub_dep_str = (obj.metadata or {}).get("Dependencies", "") or ""
                    else:
                        sub_dep_str = str(obj.get("Dependencies", obj.get("Dependencies (files)", "")) or "")
                    for sd in _parse_dependencies(sub_dep_str):
                        if sd not in queue:
                            queue.append(sd)

        return "\n\n---\n\n".join(blocks)

    
    def classify_design(self, query: str, upload_text: Optional[str] = None) -> str:
        """
        Return exactly: 'fixed design', 'group sequential design', or 'not sure'.
        """
        upload_hint = ""
        if upload_text:
            upload_hint = f"\n\n[User Upload Excerpt]\n{upload_text[:1500]}\n"

        prompt = f"""
            You are a router. Determine if the user's question is about a fixed-sample design
            or a group sequential design (as in gsDesign / gsdesign2), or if it's unclear.

            Output EXACTLY one of:
            - fixed design
            - group sequential design
            - not sure

            Guidance:
            - References to interim analyses, group sequential boundaries, spending functions,
            alpha spending, O'Brien–Fleming, Pocock, Lan–DeMets, early stopping
            => group sequential design.
            - No interim looks, single-look power/size calculations, classical fixed-sample sizing
            => fixed design.
            - If ambiguous or mixed => not sure.

            [User Query]
            {query}
            {upload_hint}

            Answer with EXACTLY one of:
            fixed design
            group sequential design
            not sure
            """
        label = self.router_llm.predict(prompt).strip()
        return _normalize_design_label(label)


    def classify_intent(self, query: str, upload_text: Optional[str] = None) -> str:
        """
        Return one of: 'code modification', 'conceptually explanation', 'not sure'.
        """
        # Keep the instruction brutally explicit and force single-line output.
        upload_hint = ""
        if upload_text:
            # Keep short to avoid wasting tokens; we only need a nudge for routing.
            sample = upload_text[:1500]
            upload_hint = f"\n\n[User Upload Excerpt]\n{sample}\n"

        prompt = f"""
            You are a router. Classify the user's request.

            Output EXACTLY one of these strings:
            - code modification
            - conceptually explanation
            - not sure

            Rules:
            - If the user asks to change, add, remove, refactor, debug, or generate code for the FIND R package, choose "code modification".
            - If the user asks for theory, methodology, intuition, how it works, pros/cons, comparisons, or examples without code changes, choose "conceptually explanation".
            - If the request is ambiguous or mixed, choose "not sure".

            [User Query]
            {query}
            {upload_hint}

            Answer with EXACTLY one of:
            code modification
            conceptually explanation
            not sure
            """
        label = self.llm_intent.predict(prompt).strip()
        return _normalize_intent_label(label)

    def smart_helper(
        self,
        query: str,
        top_k: int = 3,
        pdf_path: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify intent then route to code or package prompt automatically.
        - code modification / not sure  -> Prompt 1 (answer_code)
        - conceptually explanation      -> Prompt 2 (answer_package)
        """

        # Read upload text to append once in the prompt (not in {context})
        upload_text = read_upload_text(file_path=file_path, pdf_path=pdf_path, max_chars=60_000)

        # Router #1: design type
        design_label = self.classify_design(query=query, upload_text=upload_text)
        if design_label == "fixed design":
            design_key = "fixed"
        elif design_label == "group sequential design":
            design_key = "gs"
        else:
            design_key = "all"

        # Retrieve with the chosen main store
        docs = self.retrieve(query=query, top_k=top_k, pdf_path=pdf_path, file_path=file_path, design_key=design_key)

        # Route
        intent = self.classify_intent(query=query, upload_text=upload_text)

        if intent in ("code modification", "not sure"):
            answer = self.answer_code(query=query, docs=docs, upload_text=upload_text)
        else:  # "conceptually explanation"
            answer = self.answer_package(query=query, docs=docs, upload_text=upload_text)

        # You may include the intent in the result for debugging/UI
        result = self._format_result(answer, docs, top_k, file_path or pdf_path)
        result["intent"] = intent
        result["design_label"] = design_label
        result["design_key"] = design_key

        print("[DEBUG SMART_HELPER RESULT]", result) 
        return result

    # --------------------
    # Two distinct prompts
    # --------------------
    def answer_code(self, query: str, docs: List[Document], upload_text: Optional[str] = None) -> str:
        """
        Prompt 1: For Code Retrieval (file-level changes, minimal diffs, point edits).
        """
        context = self.build_context(docs, include_upload=False, max_deps_per_main=10, transitive=False)
        # Keep a copy for UI + parsing
        self._last_context = context

        # Build a manifest of files included in the final prompt
        manifest = []
        for block in context.split("\n\n---\n\n"):
            if not block.strip():
                continue
            head = block.splitlines()[0].strip()
            if head.startswith("[File:"):
                fname = head.split("[File:",1)[1].split("]",1)[0].strip()
                manifest.append({"type":"file","file":fname})
            elif head.startswith("[Dependency:"):
                fname = head.split("[Dependency:",1)[1].split("]",1)[0].strip()
                manifest.append({"type":"dependency","file":fname})
        self._last_manifest = manifest

        upload_block = (
        "\n\n### User Uploaded Content (raw text)\n"
        "[BEGIN USER UPLOAD]\n"
        f"{upload_text}\n"
        "[END USER UPLOAD]\n"
        if upload_text else "")

        prompt = f"""
        You are an expert AI coding assistant for the `gsDesign2` R package, which supports simulation and design of time-to-event trials under 
        proportional and non-proportional hazards, including fixed-sample and group sequential designs. It handles piecewise enrollment, failure, 
        and dropout rates, and implements testing methods such as Average Hazard Ratio (AHR), Weighted Log-Rank (WLR), and MaxCombo.

        ### Package Structure:
            The package follows a modular design:
            - define_enroll_rate.R / define_fail_rate.R: Specify enrollment, 
              event, and dropout patterns over time.
            - expected_* functions: Compute expected counts of events, 
              enrollment, and follow-up.
            - gs_design_ahr.R / gs_design_wlr.R / gs_design_maxcombo.R:
              Create group sequential designs for respective test types.
            - fixed_design_*.R: Functions for fixed-sample designs under 
              different hazard assumptions.
            - gs_power_ahr.R / gs_power_wlr.R / gs_power_maxcombo.R:
              Compute power for designs, given rates and boundaries.
            - utility_wlr.R / wlr_weight.R: Define weight functions and compute 
              test statistics and variances.
            - bounds and spending functions: Implement alpha/beta spending 
              approaches.
            - Helper functions: Shared utilities for data formatting, 
              boundary calculation, simulation support.

        **If the user also uploads a PDF file, treat its extracted text as an external source and consider BOTH the knowledge base and the external source when answering.**

        Given the user's query, the retrieved R files (with descriptions), 
        **and any external PDF content**, your job is to produce precise code 
        understanding with necessary edits.

        **Main Objectives (code retrieval & editing; dependency-aware):**
        1) **Target + dependencies.** Identify the primary file(s) AND every dependency file named in the [Dependency: …] blocks that is relevant to the requested change.
        2) **Grounding.** Use only the code in the provided [File: …] and [Dependency: …] blocks. Do NOT assume behavior outside the context.
        3) **Pinpoint edits.** Locate exact lines/blocks to modify. If a symbol/argument flows across files, trace it through dependencies.
        4) **Minimal diffs.** Propose the smallest changes that fulfill the request; keep existing APIs stable unless the user asks for a breaking change.
        5) **Cross-file consistency.** If the edit in one file implies updates in callers/callees (args, return shape, error handling), include those as separate labeled patches.
        6) **Insufficient context rule.** If a needed symbol/file isn’t present, STOP and list the exact missing file(s)/symbol(s)—do not guess.
        7) **Concise.** Show only the relevant code + brief reasoning.

        **Mandatory dependency rule:**
        - If the main file uses a symbol or parameter defined/propagated in a dependency, you MUST analyze that dependency and state whether it requires a change.
        - You MUST name each dependency you considered (by file name) and mark it **Changed** or **No change**, with a one-sentence reason.
        - **Self-check:** confirm that every dependency listed above was considered and that the diffs are minimal.

        ### User Query
        \"\"\"{query}\"\"\"{upload_block}

        Each retrieved file below includes:
        - A file name
        - Full R code with Arguments
        - A natural language description for its internal logic
        - Available Function(s)
        - Dependencies (other files it relies on)
        ### Retrieved Files (internal codebase only; dependencies inlined):
        {context}
     
        **Output (MANDATORY — do not add anything outside this block):**
        Return ONLY a single fenced block like this:

        ```sections
        [[SUMMARY]]
        ...short summary...
        [[/SUMMARY]]

        [[DEPENDENCY_IMPACT]]
        - file: dependency 1.R | status: changed | reason: ...
        - file: dependency 2.R | status: no_change | reason: ...
        - file: dependency 3.R— Changed/No change — reason...
        ** Note: List all dependencies you considered, even if they don’t change.
        [[/DEPENDENCY_IMPACT]]

        [[DIFF]]
        ```r
        # small R snippet
        (by file)Show only changed lines/blocks with a few lines of context. Prefix each block with the file name.
        ```
        [[/DIFF]]

        [[EXPLANATION]]
         ... why this is sufficient; why other deps don’t need edits (if applicable), etc.
        [[/EXPLANATION]]

        [[NEXT_STEPS]]
        - (if any)exact files/symbols required to proceed.
        [[/NEXT_STEPS]]
        
        Now respond with the best code modification or explanation.
        """
        return self.llm.predict(prompt)

    def answer_package(self, query: str, docs: List[Document], upload_text: Optional[str] = None) -> str:
        """
        Prompt 2: For Package Retrieval (architecture, API usage, behavior explanation).
        """
        context = self.build_context(docs, include_upload=False, max_deps_per_main=10, transitive=False)

                # Keep a copy for UI + parsing
        self._last_context = context

        # Build a manifest of files included in the final prompt
        manifest = []
        for block in context.split("\n\n---\n\n"):
            if not block.strip():
                continue
            head = block.splitlines()[0].strip()
            if head.startswith("[File:"):
                fname = head.split("[File:",1)[1].split("]",1)[0].strip()
                manifest.append({"type":"file","file":fname})
            elif head.startswith("[Dependency:"):
                fname = head.split("[Dependency:",1)[1].split("]",1)[0].strip()
                manifest.append({"type":"dependency","file":fname})
        self._last_manifest = manifest

        upload_block = (
        "\n\n### User Uploaded Content (raw text)\n"
        "[BEGIN USER UPLOAD]\n"
        f"{upload_text}\n"
        "[END USER UPLOAD]\n"
        if upload_text else ""
        )

        prompt = f"""
        You are an expert AI coding assistant for the `gsDesign2` R package, which supports simulation and design of time-to-event trials under 
        proportional and non-proportional hazards, including fixed-sample and group sequential designs. It handles piecewise enrollment, failure, 
        and dropout rates, and implements testing methods such as Average Hazard Ratio (AHR), Weighted Log-Rank (WLR), and MaxCombo.

        ### Package Structure:
            The package follows a modular design:
            - define_enroll_rate.R / define_fail_rate.R: Specify enrollment, 
              event, and dropout patterns over time.
            - expected_* functions: Compute expected counts of events, 
              enrollment, and follow-up.
            - gs_design_ahr.R / gs_design_wlr.R / gs_design_maxcombo.R:
              Create group sequential designs for respective test types.
            - fixed_design_*.R: Functions for fixed-sample designs under 
              different hazard assumptions.
            - gs_power_ahr.R / gs_power_wlr.R / gs_power_maxcombo.R:
              Compute power for designs, given rates and boundaries.
            - utility_wlr.R / wlr_weight.R: Define weight functions and compute 
              test statistics and variances.
            - bounds and spending functions: Implement alpha/beta spending 
              approaches.
            - Helper functions: Shared utilities for data formatting, 
              boundary calculation, simulation support.

        Each retrieved file below includes:
        - A file name
        - Full R codew ith Arguments
        - A natural language description for its internal logic
        - Available Function(s) and its parameters
        - Dependencies (other files it relies on)
        
        **If the user also uploads a PDF file, treat its extracted text as an external source and consider BOTH the knowledge base and the external source when answering.**

        **Main Objectives (for code edits; dependency-aware):**
        1) **Scope & routing.** Determine whether the request touches fixed design or group sequential pieces and identify the **primary target file(s)** AND any **required dependency files** referenced by the target’s code.
        2) **Grounding.** When the target file calls a symbol not defined within it, you **MUST** consult the corresponding **[Dependency: …]** block(s) and base your reasoning/edits on those definitions. Do not assume behavior not shown in context.
        3) **Minimal, precise diffs.** Propose the smallest change that satisfies the request. Keep existing APIs stable unless the user asks for a breaking change.
        4) **File-by-file clarity.** For each file you change, clearly label it and show only the changed lines/blocks with a few surrounding lines for context.
        5) **Cross-file consistency.** If an edit in one file requires updates in callers/callees (including argument names, returns, or error handling), include those edits as separate, labeled patches.
        6) **Tests & docs touch-ups.** If behavior changes or new params are introduced, update or add a minimal test and the roxygen/docstring snippet needed to keep the package coherent.
        7) **Insufficient context rule.** If a required function or behavior is not present in the main or dependency blocks, **stop** and list the exact missing file(s)/symbol(s) you need. Do not guess or invent code.
        8) **Token budget.** Prefer signatures + key internals from dependencies rather than full files. If context is large, summarize non-edited dependency sections (but keep exact signatures).

        **Pre-edit checklist (execute before proposing diffs):**
        - Route: fixed vs group-sequential (state which).
        - Symbols/params touched: list all occurrences of **`ratio`** across the included files (function signatures, internal calls, and where it is passed through).
        - Call chain: identify where `fixed_design_maxcombo()` calls into `gs_power_combo()` and/or `gs_design_combo()` and how `ratio` is used there.
        - Decision: which files actually need code changes for “fix allocation ratio to 1” and why (e.g., override at call sites vs changing callee defaults).
        
        **Mandatory dependency rule:**
        - If the main file uses a symbol or parameter defined/propagated in a dependency, you MUST analyze that dependency and state whether it requires a change.
        - You MUST name each dependency you considered (by file name) and mark it **Changed** or **No change**, with a one-sentence reason.

        ### User Query
        \"\"\"{query}\"\"\"{upload_block}

        ###  Retrieved Files (internal codebase only; dependencies inlined):
        [BEGIN CONTEXT]
        {context}
        [END CONTEXT]

        **Output (MANDATORY — do not add anything outside this block):**
        Return ONLY a single fenced block like this:

        ```sections
        [[CONCEPT_OVERVIEW]]
        ...short summary...
        [[/CONCEPT_OVERVIEW]]

        [[HOW_IT_WORKS]]
        ...bulleted mechanics / equations if relevant...
        [[/HOW_IT_WORKS]]

        [[WHERE_IN_PACKAGE]]
        ...functions/files from retrieval...
        [[/WHERE_IN_PACKAGE]]

        [[EXAMPLE]]
        ... small numeric or step example...
        [[/EXAMPLE]]

        [[OPTIONAL_CODE]]
        ```r
        # small R snippet

        Now respond with the best code modification or explanation.
        """

        return self.llm.predict(prompt)

    # --------------------
    # Public helpers
    # --------------------
    def code_helper(
        self,
        query: str,
        top_k: int = 3,
        pdf_path: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        docs = self.retrieve(query=query, top_k=top_k, pdf_path=pdf_path, file_path=file_path)
        upload_text = read_upload_text(file_path=file_path, pdf_path=pdf_path, max_chars=60_000)
        answer = self.answer_code(query=query, docs=docs, upload_text=upload_text)

        return self._format_result(answer, docs, top_k, file_path or pdf_path)


    def package_helper(
        self,
        query: str,
        top_k: int = 3,
        pdf_path: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        docs = self.retrieve(query=query, top_k=top_k, pdf_path=pdf_path, file_path=file_path)
        upload_text = read_upload_text(file_path=file_path, pdf_path=pdf_path, max_chars=60_000)
        answer = self.answer_package(query=query, docs=docs, upload_text=upload_text)

        return self._format_result(answer, docs, top_k, file_path or pdf_path)


    # --------------------
    # Shared result formatter
    # --------------------
    def _format_result(
        self,
        answer: str,
        docs: List[Document],
        top_k: int,
        pdf_path: Optional[str],
    ) -> Dict[str, Any]:
        preview: List[Dict[str, Any]] = []
        for rank, d in enumerate(docs, start=1):
            m = d.metadata or {}
            if m.get("is_upload") or (m.get("source") == "upload"):
                continue  # don't show uploads in Top Retrieved
            preview.append(
                {
                    "rank": rank,
                    "File": m.get("File_name"),
                    # "description_head": (m.get("description", "") or "")[:300],
                    "Docstring_summary": m.get("Docstring_summary", ""),
                    "is_upload": False,
                }
            )

        deps = []
        files_in_context = []
        try:
            for item in getattr(self, "_last_manifest", []):
                files_in_context.append(item["file"])
                if item["type"] == "dependency":
                    deps.append(item["file"])
        except Exception:
            pass

        return {
            "answer": answer,
            "retrieved": preview,
            "deps_in_context": deps,
            "files_in_context": files_in_context,
            "k": top_k,
            "used_pdf": bool(pdf_path),
        }


# =========================
# Public convenience factory
# =========================
def create_rag_helper_from_df(
    df: pd.DataFrame,
    api_key_path: Optional[str] = None,
    chat_model: str = "gpt-4o-2024-05-13",
    embedding_model: str = "text-embedding-3-large",
    intent_model: str = "gpt-4o-mini", 
    ) -> RAGHelper:
    """
    Build (llm, embedder, store) and return a ready-to-use RAGHelper.
    """

    llm_main, embedder = init_models(
    api_key_path=api_key_path,
    chat_model=chat_model,                  # main agent model
    embedding_model=embedding_model
    )

    llm_intent, _ = init_models(
        api_key_path=api_key_path,
        chat_model=intent_model,               # cheaper model for intent classification
        embedding_model=embedding_model         # embedding model ignored here
    )


    stores = build_stores_from_labeled_df(df)
    return RAGHelper(stores=stores, llm=llm_main, llm_intent=llm_intent, embedder=embedder,
                     raw_df=df, file_col="File_name", code_col="Code")

# =========================
# Example (optional)
# =========================
if __name__ == "__main__":  # pragma: no cover
    # Example usage (adjust paths/columns as needed)
    df_path = "service/gsdesign 2 with function.parquet"
    api_key_path = "service/api_key.txt"

    df = load_data(df_path)
    rag = create_rag_helper_from_df(df, api_key_path=api_key_path)

    result = rag.code_helper(
        query="Modify generate_decision_table() to include DU column.",
        top_k=3,
        pdf_path=None,  # or "uploads/some_doc.pdf"
    )
    print(result["answer"])
    for item in result["retrieved"]:
        print(item)
        
    