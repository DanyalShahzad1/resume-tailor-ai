import os
import subprocess
import tempfile
import shutil
import json
import re
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import httpx

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def extract_text_from_pdf_base64(b64_data: str) -> str:
    """Extract text from a base64-encoded PDF using pdftotext."""
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, "input.pdf")
    txt_path = os.path.join(tmpdir, "input.txt")

    with open(pdf_path, "wb") as f:
        f.write(base64.b64decode(b64_data))

    # Try pdftotext first (from poppler-utils)
    try:
        subprocess.run(
            ["pdftotext", "-layout", pdf_path, txt_path],
            capture_output=True, text=True, timeout=15,
        )
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return text
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: use Python's PyPDF2 if available
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return text.strip()
    except ImportError:
        pass

    shutil.rmtree(tmpdir, ignore_errors=True)
    return ""


def process_resume_text(raw_text: str) -> str:
    """If the resume text is a PDF base64 blob, extract text. Otherwise return as-is."""
    if raw_text.startswith("[PDF_BASE64]:"):
        b64_data = raw_text[len("[PDF_BASE64]:"):]
        extracted = extract_text_from_pdf_base64(b64_data)
        if not extracted:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF. Please upload your resume as a .tex or .txt file instead.",
            )
        return extracted
    return raw_text

LATEX_TEMPLATE = r"""
\documentclass[10pt,letterpaper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[top=0.65cm,bottom=0.65cm,left=0.9cm,right=0.9cm]{geometry}
\usepackage{titlesec}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{array}
\definecolor{primaryColor}{RGB}{0,0,0}
\hypersetup{
  colorlinks=true,
  urlcolor=primaryColor,
  pdftitle={<<NAME>> -- Resume},
  pdfauthor={<<NAME>>}
}
\raggedright
\pagestyle{empty}
\setcounter{secnumdepth}{0}
\setlength{\parindent}{0pt}
\renewcommand\labelitemi{$\vcenter{\hbox{\small$\bullet$}}$}
\titleformat{\section}{\bfseries\large}{}{0pt}{}[\titlerule]
\titlespacing*{\section}{0pt}{0.13cm}{0.07cm}
\newcommand{\baseRole}[4]{%
  \noindent\begin{tabular*}{\textwidth}{@{}p{0.74\textwidth}@{\extracolsep{\fill}}r@{}}%
  \textbf{#1} & \textbf{\textit{#2}}\\%
  \textit{#3} & \textit{#4}\\%
  \end{tabular*}\vspace{0.03cm}%
}
\newcommand{\role}[4]{\baseRole{#1}{#2}{#3}{#4}}
\newcommand{\nextrole}[4]{\vspace{0.04cm}\baseRole{#1}{#2}{#3}{#4}}
\newenvironment{highlights}
  {\begin{itemize}[topsep=0.03cm,itemsep=0.6pt,leftmargin=11pt,parsep=0pt]}
  {\end{itemize}}
\newcommand{\header}{
  \begin{center}
    {\fontsize{21pt}{21pt}\selectfont \textbf{<<NAME>>}}\\[2pt]
    <<CONTACT_LINE>>
  \end{center}
}
\begin{document}
\header
<<BODY>>
\end{document}
"""


async def call_claude(system_prompt: str, user_message: str) -> str:
    """Call Claude API to tailor the resume."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Claude API error: {response.text}")
        data = response.json()
        return data["content"][0]["text"]


def compile_latex(latex_code: str) -> str:
    """Compile LaTeX to PDF, return path to PDF file."""
    tmpdir = tempfile.mkdtemp()
    tex_path = os.path.join(tmpdir, "resume.tex")
    pdf_path = os.path.join(tmpdir, "resume.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    # Run pdflatex twice for references
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

    if not os.path.exists(pdf_path):
        log_path = os.path.join(tmpdir, "resume.log")
        log_content = ""
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_content = f.read()[-2000:]
        raise HTTPException(
            status_code=500,
            detail=f"LaTeX compilation failed. Log tail:\n{log_content}",
        )

    return pdf_path


@app.post("/api/tailor")
async def tailor_resume(
    resume_text: str = Form(...),
    job_description: str = Form(...),
):
    """Tailor the resume to match the job description using Claude."""

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server.")

    resume_text = process_resume_text(resume_text)

    system_prompt = """You are an expert resume writer and career consultant. Your job is to tailor a candidate's resume to a specific job description.

CRITICAL RULES:
1. Output ONLY valid LaTeX code for the BODY of the resume (everything between \\begin{document}/\\header and \\end{document}). Do NOT include the preamble, \\documentclass, \\usepackage, \\header command definition, \\begin{document}, or \\end{document}.
2. The resume MUST fit on exactly ONE page. Be concise. Use tight bullet points (1-2 lines each). Limit to 3-4 bullets per role. Cut less relevant roles or reduce their bullets.
3. Keep ALL facts truthful — do NOT invent experience, companies, dates, or degrees. Only rephrase/reorder/emphasize existing content.
4. Integrate keywords and phrases from the job description naturally into bullet points where the candidate genuinely has that experience.
5. Reorder sections and bullet points to front-load the most relevant experience for the target job.
6. You may adjust bullet wording to better align with the job, but never fabricate achievements or metrics.
7. Keep the Athletics section short (1 bullet max) or remove it if space is tight and it's not relevant.
8. For Technical Skills, reorder and emphasize tools/skills mentioned in the job description.

AVAILABLE LATEX COMMANDS (use these exactly):
- \\section{Section Title} — for section headers
- \\role{Title | Company}{Date Range}{Location}{Optional GPA or detail} — first role in a section
- \\nextrole{Title | Company}{Date Range}{Location}{Optional detail} — subsequent roles in same section
- \\begin{highlights} ... \\end{highlights} — bullet list environment
- \\item — each bullet point inside highlights
- For the Technical Skills section, use \\textbf{Category:} text\\\\[2pt] format

Also output a JSON block BEFORE the LaTeX with the candidate's name and contact line, formatted as:
```json
{
  "name": "Full Name",
  "contact_line": "phone \\,|\\, \\href{mailto:email}{email} \\,|\\, \\href{url}{url}"
}
```

Make sure to properly escape LaTeX special characters: $, %, &, #, _ in text content.
The $ sign in dollar amounts should be escaped as \\$.
The & in company names should be escaped as \\&.
The % sign should be escaped as \\%.
"""

    user_message = f"""Here is the candidate's current resume content:

---RESUME START---
{resume_text}
---RESUME END---

Here is the job description to tailor the resume for:

---JOB DESCRIPTION START---
{job_description}
---JOB DESCRIPTION END---

Please tailor this resume for the job. Output the JSON metadata block first, then the LaTeX body content."""

    claude_response = await call_claude(system_prompt, user_message)

    # Parse JSON metadata
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", claude_response, re.DOTALL)
    if not json_match:
        # Try without code fences
        json_match = re.search(r'\{\s*"name".*?\}', claude_response, re.DOTALL)

    name = "Candidate Name"
    contact_line = ""
    if json_match:
        try:
            metadata = json.loads(json_match.group(1) if "```" in claude_response else json_match.group(0))
            name = metadata.get("name", name)
            contact_line = metadata.get("contact_line", contact_line)
        except json.JSONDecodeError:
            pass

    # Extract LaTeX body — everything after the JSON block
    latex_body = claude_response
    if json_match:
        end_pos = json_match.end()
        # Skip past closing ``` if present
        remaining = claude_response[end_pos:]
        if remaining.lstrip().startswith("```"):
            remaining = remaining.lstrip()[3:]
        latex_body = remaining.strip()

    # Remove any ```latex fences
    latex_body = re.sub(r"```latex\s*", "", latex_body)
    latex_body = re.sub(r"```\s*$", "", latex_body)
    latex_body = re.sub(r"^```\s*", "", latex_body)

    # Build final LaTeX
    full_latex = LATEX_TEMPLATE.replace("<<NAME>>", name)
    full_latex = full_latex.replace("<<CONTACT_LINE>>", contact_line)
    full_latex = full_latex.replace("<<BODY>>", latex_body)

    # Compile to PDF
    try:
        pdf_path = compile_latex(full_latex)
    except HTTPException:
        # If compilation fails, return the LaTeX for debugging
        return JSONResponse(
            status_code=500,
            content={
                "error": "LaTeX compilation failed",
                "latex": full_latex,
                "claude_response": claude_response,
            },
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="tailored_resume.pdf",
        headers={"X-Latex-Source": "available"},
    )


@app.post("/api/tailor-json")
async def tailor_resume_json(
    resume_text: str = Form(...),
    job_description: str = Form(...),
):
    """Same as /api/tailor but returns JSON with both PDF download URL and LaTeX source."""

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server.")

    resume_text = process_resume_text(resume_text)

    system_prompt = """You are an expert resume writer and career consultant. Your job is to tailor a candidate's resume to a specific job description.

CRITICAL RULES:
1. Output ONLY valid LaTeX code for the BODY of the resume (everything between \\begin{document}/\\header and \\end{document}). Do NOT include the preamble, \\documentclass, \\usepackage, \\header command definition, \\begin{document}, or \\end{document}.
2. The resume MUST fit on exactly ONE page. Be concise. Use tight bullet points (1-2 lines each). Limit to 3-4 bullets per role. Cut less relevant roles or reduce their bullets.
3. Keep ALL facts truthful — do NOT invent experience, companies, dates, or degrees. Only rephrase/reorder/emphasize existing content.
4. Integrate keywords and phrases from the job description naturally into bullet points where the candidate genuinely has that experience.
5. Reorder sections and bullet points to front-load the most relevant experience for the target job.
6. You may adjust bullet wording to better align with the job, but never fabricate achievements or metrics.
7. Keep the Athletics section short (1 bullet max) or remove it if space is tight and it's not relevant.
8. For Technical Skills, reorder and emphasize tools/skills mentioned in the job description.

AVAILABLE LATEX COMMANDS (use these exactly):
- \\section{Section Title} — for section headers
- \\role{Title | Company}{Date Range}{Location}{Optional GPA or detail} — first role in a section
- \\nextrole{Title | Company}{Date Range}{Location}{Optional detail} — subsequent roles in same section
- \\begin{highlights} ... \\end{highlights} — bullet list environment
- \\item — each bullet point inside highlights
- For the Technical Skills section, use \\textbf{Category:} text\\\\[2pt] format

Also output a JSON block BEFORE the LaTeX with the candidate's name and contact line, formatted as:
```json
{
  "name": "Full Name",
  "contact_line": "phone \\,|\\, \\href{mailto:email}{email} \\,|\\, \\href{url}{url}"
}
```

Make sure to properly escape LaTeX special characters: $, %, &, #, _ in text content.
The $ sign in dollar amounts should be escaped as \\$.
The & in company names should be escaped as \\&.
The % sign should be escaped as \\%.
"""

    user_message = f"""Here is the candidate's current resume content:

---RESUME START---
{resume_text}
---RESUME END---

Here is the job description to tailor the resume for:

---JOB DESCRIPTION START---
{job_description}
---JOB DESCRIPTION END---

Please tailor this resume for the job. Output the JSON metadata block first, then the LaTeX body content."""

    claude_response = await call_claude(system_prompt, user_message)

    # Parse JSON metadata
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", claude_response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{\s*"name".*?\}', claude_response, re.DOTALL)

    name = "Candidate Name"
    contact_line = ""
    if json_match:
        try:
            metadata = json.loads(json_match.group(1) if "```" in claude_response else json_match.group(0))
            name = metadata.get("name", name)
            contact_line = metadata.get("contact_line", contact_line)
        except json.JSONDecodeError:
            pass

    # Extract LaTeX body
    latex_body = claude_response
    if json_match:
        end_pos = json_match.end()
        remaining = claude_response[end_pos:]
        if remaining.lstrip().startswith("```"):
            remaining = remaining.lstrip()[3:]
        latex_body = remaining.strip()

    latex_body = re.sub(r"```latex\s*", "", latex_body)
    latex_body = re.sub(r"```\s*$", "", latex_body)
    latex_body = re.sub(r"^```\s*", "", latex_body)

    full_latex = LATEX_TEMPLATE.replace("<<NAME>>", name)
    full_latex = full_latex.replace("<<CONTACT_LINE>>", contact_line)
    full_latex = full_latex.replace("<<BODY>>", latex_body)

    # Compile to PDF
    pdf_base64 = None
    compile_error = None
    try:
        pdf_path = compile_latex(full_latex)
        import base64
        with open(pdf_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
    except HTTPException as e:
        compile_error = str(e.detail)

    return JSONResponse(content={
        "success": pdf_base64 is not None,
        "pdf_base64": pdf_base64,
        "latex": full_latex,
        "error": compile_error,
    })


# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
