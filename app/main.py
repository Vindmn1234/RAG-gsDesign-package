# app/main.py
from unittest import result
from fastapi import FastAPI, HTTPException, Query, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import traceback
from service.langchain_news import load_data, create_rag_helper_from_df, first_sentence, html_escape
from service.utils_logging import append_jsonl
import re
from fastapi import UploadFile, File
from pathlib import Path
import shutil
import os
from fastapi.staticfiles import StaticFiles

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = "service/conversations.jsonl"

app = FastAPI(title="News Processing API")

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


def save_upload_file(upload: UploadFile) -> str:
    """
    Save uploaded PDF/TXT and return absolute path.
    """
    filename = upload.filename or "uploaded"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".pdf", ".txt"}:
        raise ValueError("Only .pdf and .txt are supported.")

    dest = (UPLOAD_DIR / filename).resolve()
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    
    return str(dest)

# Global RAG helper (initialized at startup)
rag = None


@app.on_event("startup")
async def startup_event():
    """
    Load the pre-embedded table and initialize the RAG helper.
    """
    global rag
    try:
        # Adjust to your actual data path
        df = load_data("service/gsdesign 2 with function.parquet")
        rag = create_rag_helper_from_df(
            df,
            api_key_path="service/api_key.txt",  # or rely on OPENAI_API_KEY env var
            chat_model="gpt-4o-2024-05-13",
            embedding_model="text-embedding-3-large",
            intent_model="gpt-4o-mini",
        )
        print("RAG initialized.")
    except Exception as e:
        # Don't crash app startup; just log the error.
        print(f"Error during startup data processing: {e}")

@app.get("/", response_class=HTMLResponse)
# Favicon image source: https://www.favicon.cc/?action=icon&file_id=819961
# Banner image source: https://www.shutterstock.com/search/latest-news-banner
def welcome():
    """
    A welcome page with a favicon, banner image, and detailed introduction.
    """
    html_content = """
        <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Welcome to News Processing API</title>

      <!-- Favicon -->
      <link rel="icon" href="/static/favicon.ico" type="image/x-icon">

      <style>
        /* ---- Page-level ---- */
        body {
          background-color: #f2f2f2;
          font-family: Arial, sans-serif;
          text-align: center;
          padding: 50px;
          margin: 0;
        }
        /* Wrapper so the logo can live outside the white card */
        .page-wrapper {
          position: relative;
          display: inline-block;   /* keeps wrapper width = card width */
        }

        /* ---- Card ---- */
        .container {
          background-color: #fff;
          padding: 30px 30px 40px;
          border-radius: 8px;
          box-shadow: 0 0 10px rgba(0,0,0,0.1);
          max-width: 800px;
        }

        /* ---- Top-right logo ---- */
        .logo {
          position: absolute;
          top: 12px;
          right: 8px;
          width: 110px;      
          height: auto;
        }

        /* ---- Typography ---- */
        h1 {
          color: #333;
          margin-bottom: 20px;
        }
        p {
          color: #555;
          line-height: 1.6;
          margin-bottom: 20px;
        }

        /* ---- CTA button ---- */
        .button {
          background-color: #007BFF;
          color: #fff;
          padding: 10px 20px;
          text-decoration: none;
          border-radius: 5px;
          font-size: 16px;
          display: inline-block;
          margin-top: 20px;
        }
        .button:hover {
          background-color: #0056b3;
        }
      </style>
    </head>

    <body>
      <div class="page-wrapper">

        <!-- Small logo in the corner -->
        <img src="/static/logo.jpg" alt="FIND logo" class="logo">

        <!-- Main content card -->
        <div class="container">
          <h1>Welcome to the FIND R Package AI Assistant</h1>

          <p>Accelerate your clinical trial simulation workflow</p>

          <p>
            This web assistant uses a Retrieval-Augmented Generation (RAG) framework built on
            LangChain to help you explore, query, and understand the <strong>gsDesign2</strong> R
            package — a modular toolkit for for deriving fixed and group sequential designs under non-proportional hazards.
          </p>

          <p>
           Whether you’re designing a fixed-sample study or a group sequential trial with <strong>gsDesign2</strong>, our assistant can retrieve specific 
           function logic (e.g., <code>gs_design()</code>, <code>gs_info_combo()</code>, <code>utility_wlr()</code>, <code>wlr_weight()</code>), 
           compute and explain boundaries/spending (O’Brien–Fleming, Pocock, Lan–DeMets), and clarify power/size and operating characteristics. 
           It uses vector search over the gsDesign2 codebase (plus your uploads) and LLM reasoning to provide concise explanations or minimal code edits—whichever your query needs.
          </p>

          <p>
            Upload relevant R scripts or documentation, ask natural-language questions, and
            receive code-aware, context-rich answers tailored to your simulation needs.
          </p>

          <a class="button" href="/interface">Start Exploring</a>
        </div>
      </div>
    </body>
    </html>

    """
    return HTMLResponse(content=html_content)

# @app.get("/status", response_class=HTMLResponse)
# def status():
#     """
#     Check whether the global variables have been initialized.
#     """
#     if vector_db and qa_chain:
#         return "<h2>Data is processed. The vector store and QA chain are initialized.</h2>"
#     else:
#         return "<h2>Data is not processed yet. Please process the data first.</h2>"

@app.get("/interface", response_class=HTMLResponse)
def query_interface():
    html_content = """
    <html>
      <head>
        <title>News Query Interface</title>
        <style>
          body { background-color: #f2f2f2; font-family: Arial, sans-serif; margin: 0; padding: 0; }
          .container { max-width: 1200px; margin: 20px auto; background-color: #fff; padding: 30px;
                       border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
          h1, h2 { text-align: center; color: #333; }
          p { font-size: 16px; line-height: 1.6; color: #555; margin-bottom: 20px; }
          form { display: flex; flex-direction: column; }
          label { margin-top: 15px; font-weight: bold; }
          select, textarea, input[type="file"] { padding: 10px; font-size: 16px; margin-top: 5px; }
          input[type="submit"] { margin-top: 20px; padding: 10px; font-size: 16px; background-color: #007BFF; 
                                 color: #fff; border: none; border-radius: 5px; cursor: pointer; }
          input[type="submit"]:hover { background-color: #0056b3; }
          .back-link { display: inline-block; margin-top: 20px; text-decoration: none; color: #007BFF; font-weight: bold; }
          .back-link:hover { text-decoration: underline; }
          .hint { font-size: 13px; color: #777; }
          iframe { border: none; width: 100%; height: 600px; margin-bottom: 20px; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>gsDesign2 Package Query Interface</h1>

          <!-- Function dependency network iframe -->
          <iframe src="/static/pkgnet_report.html"></iframe>

          <!-- Query form -->
          <form action="/process-query" method="post" enctype="multipart/form-data">
            <label for="query_text"><strong>Enter Your Query:</strong></label>
            <textarea id="query_text" name="query_text" rows="5" cols="60"
              placeholder="e.g., How does the choice of a spending function influence the stopping boundaries in a group sequential trial?"></textarea>

            <label for="pdf_file"><strong>Upload a PDF or text file(optional):</strong></label>
            <input type="file" id="pdf_file" name="pdf_file" accept="application/pdf,.pdf,text/plain,.txt">
            <span class="hint">If provided, we will augment retrieval with your PDF.</span>

            <input type="submit" value="Submit Query" class="button">
          </form>

          <br>
          <a class="back-link" href="/">&#8592; Back to Home</a>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

import re
from html import escape as _esc

def _html(s: str) -> str:
    return _esc(s or "", quote=True)

def _crop_sections_block(text: str) -> str:
    """
    If the answer is wrapped with ```sections ... ```, return the INNER region
    from the first ```sections to the LAST ``` after it. Otherwise return full text.
    """
    if not text:
        return ""
    m = re.search(r"```sections\b", text, flags=re.I)
    if not m:
        return text
    start = m.end()
    end = text.rfind("```")
    return text[start:end] if end > start else text

# Exact-name backreference: [[NAME]] ... [[/NAME]]
PAIR_RE = re.compile(r"\[\[([^\]]+?)\]\](.*?)\[\[/\1\]\]", flags=re.S)

def parse_tag_blocks(answer_text: str) -> list[dict]:
    """
    Returns an ordered list of blocks:
      [{"tag_raw": "DIFF", "content": "..."} , ...]
    No assumptions about tag names; works with NEXT-STEPS, EXPLANATION, etc.
    """
    s = _crop_sections_block(answer_text or "")
    blocks = []
    for m in PAIR_RE.finditer(s):
        tag = (m.group(1) or "").strip()           # e.g., DIFF, EXPLANATION, NEXT-STEPS
        content = (m.group(2) or "").strip()
        blocks.append({"tag_raw": tag, "content": content})
    # Debug: see what we found
    print("[SECTIONS FOUND]", [b["tag_raw"] for b in blocks])
    return blocks

def _render_text_with_code(text: str) -> str:
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    html = []
    for part in parts:
        if part.startswith("```"):
            m = re.match(r"```([a-zA-Z0-9_-]*)\s*(.*?)\s*```", part, flags=re.S)
            lang = (m.group(1) or "").strip() if m else ""
            code = (m.group(2) if m else part.strip("`")).rstrip()
            html.append(f'<pre class="code"><code class="language-{_html(lang)}">{_html(code)}</code></pre>')
        else:
            chunk = part.strip()
            if not chunk:
                continue
            lines = chunk.splitlines()
            buf, ul = [], []
            def flush_buf():
                if buf:
                    html.append("<p>" + _html("\n".join(buf)).replace("\n", "<br>") + "</p>")
                    buf.clear()
            def flush_ul():
                if ul:
                    html.append("<ul>" + "".join(f"<li>{_html(li.strip()[2:])}</li>" for li in ul) + "</ul>")
                    ul.clear()
            for ln in lines:
                if ln.lstrip().startswith("- "):
                    flush_buf(); ul.append(ln)
                else:
                    flush_ul(); buf.append(ln)
            flush_ul(); flush_buf()
    return "".join(html)

def _pretty_title(raw: str) -> str:
    return raw[:1].upper() + raw[1:]

def render_sections_dynamic(answer_text: str) -> str:
    blocks = parse_tag_blocks(answer_text)
    if not blocks:
        return "<p><em>No structured sections found.</em></p>"
    cards = []
    for b in blocks:
        raw = b["tag_raw"]
        content = b["content"]
        title = _pretty_title(raw)

        # Special readability tweak for DIFF: if no fences, wrap as code
        if raw.strip().upper() == "DIFF" and "```" not in content:
            body = f'<pre class="code"><code class="language-r">{_html(content)}</code></pre>'
        else:
            body = _render_text_with_code(content)

        cards.append(
            f"""
            <section class="card">
              <div class="card__header">{_html(title)}</div>
              <div class="card__body">{body}</div>
            </section>
            """
        )
    return "\n".join(cards)

@app.post("/process-query", response_class=HTMLResponse)
def process_query(
    query_text: str = Form(...),
    pdf_file: UploadFile | None = File(None),  # optional upload (.pdf or .txt)
):
    if rag is None:
        return HTMLResponse(
            content="<h2>Error: Data not processed yet. Please refresh after startup completes.</h2>",
            status_code=400,
        )

    try:
        file_path = None
        if pdf_file and pdf_file.filename:
            try:
                file_path = save_upload_file(pdf_file)  # must accept .pdf AND .txt
            finally:
                try:
                    pdf_file.file.close()  # since this endpoint is sync
                except Exception:
                    pass

        # 🔁 Route automatically (no query_style anymore)
        result = rag.smart_helper(
            query=query_text,
            top_k=1,
            file_path=file_path,   # works for both .pdf and .txt
        )

        # ✅ log
        try:
            append_jsonl(
                LOG_PATH,
                {
                    "user_input": query_text,
                    "llm_response": result.get("answer", ""),
                    "used_upload": result.get("used_pdf", False),  # backend sets True if any upload used
                    "design" : result.get("design_label", ""),
                    "intent": result.get("intent", ""),
                    "top_k": result.get("k", 1),
                    "retrieved_preview": result.get("retrieved", []),
                },
            )
        except Exception as log_err:
            print(f"[LOGGING] Failed to append JSONL: {log_err}")

        # ✅ format UI
        design = result.get("design_label", "")
        intent = result.get("intent", "")
        used_upload = "Yes" if result.get("used_pdf") else "No"
        # answer_html = (result.get("answer", "") or "").replace("\n", "<br>")
        answer_text = result.get("answer", "") or ""
        sections_html = render_sections_dynamic(answer_text)

        retrieved_rows_all = result.get("retrieved", []) or []
        retrieved_rows = [r for r in retrieved_rows_all if not r.get("is_upload")]  # internal only

        retrieved_html = "<ul>" + "".join(
            f"<li><strong>#{row.get('rank')}</strong> "
            f"{html_escape(row.get('file',''))} —"
            f" {html_escape(row.get('Docstring_summary','') or '')}</li>"
            for row in retrieved_rows
            ) + "</ul>"
        
        # NEW: Dependencies that were actually included in the prompt
        deps = result.get("deps_in_context") or []
        deps_html = "" if not deps else (
            "<div><strong>Dependencies included:</strong><ul>" +
            "".join(f"<li>{html_escape(d)}</li>" for d in deps) +
            "</ul></div>")

        html_response = f"""
        <html>
          <head>
            <title>Query Result</title>
            <link rel="stylesheet" href="/static/styles.css">
            <style>
              /* Limit page width and center */
              body {{
                display: flex;
                justify-content: center;
                padding: 20px;
                background: #f9f9f9;
              }}
              .container {{
                max-width: 900px;
                width: 100%;
                background: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
              }}

              .card {{ border:1px solid #eee; border-radius:10px; margin:14px 0; overflow:hidden; }}
              .card__header {{ background:#fafafa; padding:10px 14px; font-weight:600; color:#222; border-bottom:1px solid #eee; }}
              .card__body {{ padding:14px; line-height:1.55; color:#333; }}
              .card__body ul {{ margin:0.5rem 0 0.5rem 1.25rem; }}
              .code {{ background:#0f172a; color:#e2e8f0; padding:12px; border-radius:8px; overflow:auto; }}
              .code code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 0.92rem; }}
            </style>
          </head>
          <body>
            <div class="container">
              <h1>Query Result</h1>
              <div class="meta">
                <p><strong>Detected Design:</strong> {html_escape(design)}</p>
                <p><strong>Detected Intent:</strong> {html_escape(intent)}</p>
                <p><strong>Used Upload:</strong> {used_upload}</p>
                <p><strong>Query:</strong> {html_escape(query_text)}</p>
              </div>

              <div class="section">
                <h2>Response</h2>
                <div>{sections_html}</div>
              </div>

              <div class="section">
                <h2>Top Retrieved (internal codebase)</h2>
                {retrieved_html}
                {deps_html}
              </div>

              <div class="section">
                <a href="/interface">Back to Query Interface</a>
              </div>
            </div>
          </body>
        </html>
        """

        return HTMLResponse(content=html_response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(
            content=f"<h2>Error processing query: {e}</h2>",
            status_code=500,
        )