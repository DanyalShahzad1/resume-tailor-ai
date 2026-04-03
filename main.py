import os
import subprocess
import tempfile
import shutil
import json
import re
import base64
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import httpx

app = FastAPI()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ── PDF text extraction ──────────────────────────────────────────────
def extract_text_from_pdf_base64(b64_data: str) -> str:
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, "input.pdf")
    txt_path = os.path.join(tmpdir, "input.txt")
    with open(pdf_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    try:
        subprocess.run(["pdftotext", "-layout", pdf_path, txt_path],
                       capture_output=True, text=True, timeout=15)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return text
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return text.strip()
    except Exception:
        pass
    shutil.rmtree(tmpdir, ignore_errors=True)
    return ""


def process_resume_text(raw_text: str) -> str:
    if raw_text.startswith("[PDF_BASE64]:"):
        b64_data = raw_text[len("[PDF_BASE64]:"):]
        extracted = extract_text_from_pdf_base64(b64_data)
        if not extracted:
            raise ValueError("Could not extract text from PDF. Upload as .tex or .txt instead.")
        return extracted
    return raw_text


# ── LaTeX template ───────────────────────────────────────────────────
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
  pdftitle={<<n>> -- Resume},
  pdfauthor={<<n>>}
}
\raggedright
\sloppy
\hyphenpenalty=10000
\exhyphenpenalty=10000
\pagestyle{empty}
\setcounter{secnumdepth}{0}
\setlength{\parindent}{0pt}
\renewcommand\labelitemi{$\vcenter{\hbox{\small$\bullet$}}$}
\titleformat{\section}{\bfseries\large}{}{0pt}{}[\titlerule]
\titlespacing*{\section}{0pt}{0.13cm}{0.07cm}
\newcommand{\baseRole}[4]{%
  \noindent\begin{tabular*}{\textwidth}{@{}p{0.64\textwidth}@{\extracolsep{\fill}}r@{}}%
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
    {\fontsize{21pt}{21pt}\selectfont \textbf{<<n>>}}\\[2pt]
    <<CONTACT_LINE>>
  \end{center}
}
\begin{document}
\header
<<BODY>>
\end{document}
"""


# ── Prompts ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = r"""You are an elite resume tailoring expert. Your job is to aggressively rewrite a candidate's resume bullets to directly mirror a job description — while keeping the same underlying experiences and facts.

WHAT YOU DO:
- Read the job description and deeply understand what they're looking for.
- REWRITE each bullet point so it reads like the candidate was doing exactly what the job description asks for.
- Use the job description's EXACT phrases, terminology, and action verbs throughout.
- Front-load each bullet with the most job-relevant keyword or phrase.
- Reframe accomplishments to emphasize aspects most relevant to the target job.
- Rewrite Technical Skills categories and ordering to mirror the job description's language.

WHAT YOU MUST KEEP THE SAME:
- Same sections, same roles, same number of bullet points per role. Do NOT add or remove bullets.
- All facts, numbers, metrics, percentages, dates, and company names stay unchanged.
- Same order of roles and sections as the original.

PAGE FILLING (CRITICAL):
- Make each bullet 10-15 words LONGER than the original. Add job-relevant keywords, context, outcomes, and detail.
- You MUST generate content that OVERFLOWS past 1 page. Aim for roughly 1.15-1.25 pages of content. The system will automatically compress spacing to fit it perfectly on exactly 1 page.
- Do NOT try to fit on 1 page yourself. Write MORE than fits. The system handles the fitting.
- If a bullet was 1 line, make it 1.5-2 lines. If it was 2 lines, make it 2-2.5 lines.
- It is MUCH better to write too much (system shrinks it) than too little (leaves blank space).

CRITICAL OUTPUT FORMAT:
1. First output a JSON block with name and contact info:
```json
{"name": "Full Name", "contact_line": "phone \\,|\\, \\href{mailto:email}{email} \\,|\\, \\href{url}{url}"}
```

2. Then output ONLY the LaTeX BODY content. Do NOT include \documentclass, \usepackage, \begin{document}, \end{document}, or \header.

LATEX COMMANDS TO USE:
- \section{Title}
- \role{Title | Company}{Dates}{Location}{}  (first entry in section)
- \nextrole{Title | Company}{Dates}{Location}{}  (subsequent entries)
- \begin{highlights} \item bullet text \end{highlights}
- \textbf{Category:} text\\[2pt]  (for Technical Skills)

ESCAPING RULES (CRITICAL):
- Dollar amounts: \$20M
- Ampersand: FP\&A, SG\&A
- Percent: 15\%
- Hash: \#
"""

SYSTEM_PROMPT_2PAGE = r"""You are an elite resume tailoring expert. Your job is to aggressively rewrite a candidate's resume bullets to directly mirror a job description — while condensing a long resume into a focused 2-page version.

Use the EXACT SAME formatting and LaTeX commands as a 1-page resume — just with more content to fill 2 full pages.

WHAT YOU DO:
- Read the job description and deeply understand what they're looking for.
- REWRITE each bullet point so it reads like the candidate was doing exactly what the job description asks for.
- Use the job description's EXACT phrases, terminology, and action verbs throughout.
- For very experienced candidates, focus on the last 10-15 years. Older roles get fewer bullets.
- You MAY remove roles or sections that are completely irrelevant to the target job.
- Keep roles in reverse chronological order within each section.

CONTENT AMOUNT:
- Most recent/relevant roles: 4-5 bullet points each.
- Older or less relevant roles: 2-3 bullet points each.
- Education: degrees, institutions, dates only — use \role format, no bullets.
- Technical Skills: 2-3 category lines using \textbf{Category:} format.
- Certifications: list them if present.

PAGE FILLING (CRITICAL):
- You MUST generate content that OVERFLOWS past 2 pages. Aim for roughly 2.2-2.4 pages of content.
- The system will automatically compress spacing to fit exactly 2 pages.
- Do NOT try to fit on 2 pages yourself. Write MORE than fits. The system handles the fitting.
- Use 5-6 bullet points per major role, each 1.5-2 lines long.
- Use 3-4 bullet points per minor role.
- It is MUCH better to write too much (system shrinks it) than too little (leaves blank space).

WHAT YOU MUST KEEP:
- All facts, numbers, metrics, percentages, dates, and company names must be truthful.
- Never invent or fabricate experience.
- The 4th parameter of \role and \nextrole must be SHORT (max 3 words) or empty {}.

CRITICAL OUTPUT FORMAT:
1. First output a JSON block with name and contact info:
```json
{"name": "Full Name", "contact_line": "phone \\,|\\, \\href{mailto:email}{email} \\,|\\, \\href{url}{url}"}
```

2. Then output ONLY the LaTeX BODY content. Do NOT include \documentclass, \usepackage, \begin{document}, \end{document}, or \header.

LATEX COMMANDS TO USE (same as 1-page):
- \section{Title}
- \role{Title | Company}{Dates}{Location}{}
- \nextrole{Title | Company}{Dates}{Location}{}
- \begin{highlights} \item bullet text \end{highlights}
- \textbf{Category:} text\\[2pt]

ESCAPING RULES (CRITICAL):
- Dollar amounts: \$20M
- Ampersand: FP\&A, SG\&A
- Percent: 15\%
- Hash: \#
"""


# ── Claude API ───────────────────────────────────────────────────────
async def call_claude(system_prompt: str, user_message: str) -> str:
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
                "max_tokens": 8096,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
        )
        if response.status_code != 200:
            raise Exception(f"Claude API error ({response.status_code}): {response.text[:500]}")
        data = response.json()
        return data["content"][0]["text"]


# ── PDF page count ───────────────────────────────────────────────────
def get_pdf_page_count(pdf_path: str) -> int:
    """Get page count using PyPDF2 (guaranteed installed)."""
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        pass
    try:
        result = subprocess.run(["pdfinfo", pdf_path],
                                capture_output=True, text=True, timeout=10)
        for line in result.stdout.split("\n"):
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 1


# ── LaTeX compilation ────────────────────────────────────────────────
def compile_latex(latex_code: str) -> str:
    tmpdir = tempfile.mkdtemp()
    tex_path = os.path.join(tmpdir, "resume.tex")
    pdf_path = os.path.join(tmpdir, "resume.pdf")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
            capture_output=True, text=True, timeout=30,
        )
    if not os.path.exists(pdf_path):
        log_path = os.path.join(tmpdir, "resume.log")
        log_content = ""
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_content = f.read()[-2000:]
        raise Exception(f"LaTeX compilation failed. Log:\n{log_content}")
    return pdf_path


def compile_latex_fit_pages(latex_code: str, max_pages: int = 1) -> str:
    """Compile LaTeX and auto-adjust to fit exactly max_pages."""
    pdf_path = compile_latex(latex_code)
    pages = get_pdf_page_count(pdf_path)

    # === SHRINK if over max_pages ===
    if pages > max_pages:
        # Level 1: tighten all spacing
        m = latex_code
        m = m.replace(r"\itemsep=0.6pt", r"\itemsep=0pt")
        m = m.replace(r"\titlespacing*{\section}{0pt}{0.13cm}{0.07cm}",
                       r"\titlespacing*{\section}{0pt}{0.08cm}{0.04cm}")
        m = m.replace(r"\vspace{0.04cm}", r"\vspace{0.01cm}")
        m = m.replace(r"\end{tabular*}\vspace{0.03cm}", r"\end{tabular*}\vspace{0.01cm}")
        m = m.replace(r"topsep=0.03cm", r"topsep=0.01cm")
        shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
        pdf_path = compile_latex(m)
        if get_pdf_page_count(pdf_path) <= max_pages:
            return pdf_path

        # Level 2: tighten margins
        m = m.replace(r"top=0.65cm,bottom=0.65cm", r"top=0.4cm,bottom=0.4cm")
        m = m.replace(r"left=0.9cm,right=0.9cm", r"left=0.7cm,right=0.7cm")
        shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
        pdf_path = compile_latex(m)
        if get_pdf_page_count(pdf_path) <= max_pages:
            return pdf_path

        # Level 3: \small font
        m = m.replace(r"\begin{document}", r"\begin{document}\small")
        shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
        pdf_path = compile_latex(m)
        if get_pdf_page_count(pdf_path) <= max_pages:
            return pdf_path

        # Level 4: \footnotesize font
        m = m.replace(r"\begin{document}\small", r"\begin{document}\footnotesize")
        shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
        pdf_path = compile_latex(m)
        return pdf_path

    # === EXPAND if under max_pages ===
    if pages < max_pages:
        expand_steps = [
            (r"\itemsep=0.6pt", r"\itemsep=2.5pt"),
            (r"\titlespacing*{\section}{0pt}{0.13cm}{0.07cm}",
             r"\titlespacing*{\section}{0pt}{0.25cm}{0.12cm}"),
            (r"\vspace{0.04cm}", r"\vspace{0.12cm}"),
            (r"\end{tabular*}\vspace{0.03cm}", r"\end{tabular*}\vspace{0.08cm}"),
        ]
        m = latex_code
        for old, new in expand_steps:
            test = m.replace(old, new)
            tp = compile_latex(test)
            if get_pdf_page_count(tp) <= max_pages:
                m = test
                shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
                pdf_path = tp
            else:
                shutil.rmtree(os.path.dirname(tp), ignore_errors=True)
        return pdf_path

    return pdf_path


# ── Response parsing & sanitization ──────────────────────────────────
def parse_claude_response(claude_response: str):
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", claude_response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{\s*"name".*?\}', claude_response, re.DOTALL)
    name = "Candidate Name"
    contact_line = ""
    if json_match:
        try:
            raw = json_match.group(1) if json_match.lastindex else json_match.group(0)
            metadata = json.loads(raw)
            name = metadata.get("name", name)
            contact_line = metadata.get("contact_line", contact_line)
        except json.JSONDecodeError:
            pass
    latex_body = claude_response
    if json_match:
        end_pos = json_match.end()
        remaining = claude_response[end_pos:]
        remaining = re.sub(r"^\s*```", "", remaining)
        latex_body = remaining.strip()
    latex_body = re.sub(r"```latex\s*", "", latex_body)
    latex_body = re.sub(r"```\s*$", "", latex_body)
    latex_body = re.sub(r"^```\s*", "", latex_body)
    return name, contact_line, latex_body.strip()


def sanitize_latex(body: str) -> str:
    body = re.sub(r"\\begin\{document\}", "", body)
    body = re.sub(r"\\end\{document\}", "", body)
    body = re.sub(r"\\documentclass.*\n?", "", body)
    body = re.sub(r"\\usepackage.*\n?", "", body)
    body = re.sub(r"^\\header\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r'(?<!\\)\$(\d)', r'\\$\1', body)
    body = re.sub(r'(?<!\\)%', r'\\%', body)
    lines = body.split("\n")
    fixed = []
    for line in lines:
        if line.strip().startswith("\\item"):
            line = re.sub(r'(?<!\\)&', r'\\&', line)
        fixed.append(line)
    body = "\n".join(fixed)
    body = body.replace("\\\\&", "\\&")
    body = body.replace("\\\\%", "\\%")
    body = re.sub(r'\\\\\$(\d)', r'\\$\1', body)
    return body


def build_full_latex(name: str, contact_line: str, body: str) -> str:
    full = LATEX_TEMPLATE.replace("<<n>>", name)
    full = full.replace("<<CONTACT_LINE>>", contact_line)
    full = full.replace("<<BODY>>", body)
    return full


# ── API Endpoints ────────────────────────────────────────────────────
@app.post("/api/check-resume")
async def check_resume(resume_text: str = Form(...)):
    try:
        if resume_text.startswith("[PDF_BASE64]:"):
            b64_data = resume_text[len("[PDF_BASE64]:"):]
            tmpdir = tempfile.mkdtemp()
            pdf_path = os.path.join(tmpdir, "check.pdf")
            with open(pdf_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            pages = get_pdf_page_count(pdf_path)
            shutil.rmtree(tmpdir, ignore_errors=True)
            return JSONResponse(content={"pages": pages})
        else:
            char_count = len(resume_text.strip())
            estimated_pages = max(1, min(int(char_count / 2800) + 1, 10)) if char_count > 4000 else 1
            return JSONResponse(content={"pages": estimated_pages})
    except Exception:
        return JSONResponse(content={"pages": 1})


@app.post("/api/tailor-json")
async def tailor_resume_json(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    max_pages: int = Form(1),
):
    try:
        if not ANTHROPIC_API_KEY:
            return JSONResponse(content={
                "success": False, "pdf_base64": None, "latex": "",
                "error": "ANTHROPIC_API_KEY not configured. Add it in Railway Variables."
            })

        max_pages = max(1, min(2, max_pages))
        resume_text = process_resume_text(resume_text)
        prompt = SYSTEM_PROMPT if max_pages == 1 else SYSTEM_PROMPT_2PAGE

        user_message = f"""Here is the candidate's current resume:

---RESUME START---
{resume_text}
---RESUME END---

Here is the target job description:

---JOB DESCRIPTION START---
{job_description}
---JOB DESCRIPTION END---

Tailor this resume for the job as a {max_pages}-page resume. Output the JSON metadata block first, then the LaTeX body."""

        claude_response = await call_claude(prompt, user_message)
        name, contact_line, latex_body = parse_claude_response(claude_response)
        latex_body = sanitize_latex(latex_body)
        full_latex = build_full_latex(name, contact_line, latex_body)

        pdf_base64_str = None
        compile_error = None
        try:
            pdf_path = compile_latex_fit_pages(full_latex, max_pages)
            with open(pdf_path, "rb") as f:
                pdf_base64_str = base64.b64encode(f.read()).decode("utf-8")
        except Exception as first_err:
            try:
                fix_msg = f"""This LaTeX body failed to compile:

{latex_body[:3000]}

Error log:
{str(first_err)[-1500:]}

Fix it so it compiles. Output ONLY the corrected LaTeX body starting from \\section. No code fences, no explanation."""

                fixed = await call_claude(
                    "You fix broken LaTeX. Output ONLY corrected LaTeX code. No markdown, no explanation, no code fences.",
                    fix_msg,
                )
                fixed = re.sub(r"```.*?\n?", "", fixed).strip()
                fixed = sanitize_latex(fixed)
                full_latex = build_full_latex(name, contact_line, fixed)
                pdf_path = compile_latex_fit_pages(full_latex, max_pages)
                with open(pdf_path, "rb") as f:
                    pdf_base64_str = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                compile_error = f"Compilation failed after auto-fix retry. Error: {str(first_err)[-500:]}"

        return JSONResponse(content={
            "success": pdf_base64_str is not None,
            "pdf_base64": pdf_base64_str,
            "latex": full_latex,
            "error": compile_error,
        })

    except ValueError as e:
        return JSONResponse(content={"success": False, "pdf_base64": None, "latex": "", "error": str(e)})
    except Exception as e:
        return JSONResponse(content={"success": False, "pdf_base64": None, "latex": "", "error": f"Server error: {str(e)}"})


app.mount("/", StaticFiles(directory="static", html=True), name="static")
