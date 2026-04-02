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

SYSTEM_PROMPT = r"""You are an expert resume writer who outputs compilable LaTeX. Tailor a candidate's resume to a specific job description.

YOUR APPROACH:
- Keep the SAME structure: same sections, same number of roles, same number of bullet points per role.
- Do NOT add or remove any roles, sections, or bullet points.
- You SHOULD meaningfully reword each bullet point to emphasize skills, tools, and outcomes that match the job description.
- You CAN rephrase entire sentences — just keep the same underlying facts, metrics, and experiences.
- Weave in keywords and terminology from the job description naturally where the candidate has genuine experience.
- Reorder bullet points WITHIN each role to put the most job-relevant ones first.
- In Technical Skills, reorder tools to put job-relevant ones first, and add any job-mentioned tools the candidate actually knows.
- Keep each bullet approximately the same length as the original (do not make them significantly longer or shorter).

CHRONOLOGICAL ORDER (CRITICAL):
- Within each section, roles MUST be in reverse chronological order (most recent date first).
- Education: most recent degree first.
- Experience: most recent job first.
- Projects & Leadership: most recent first.

CRITICAL OUTPUT FORMAT:
1. First output a JSON block with name and contact info:
```json
{"name": "Full Name", "contact_line": "phone \\,|\\, \\href{mailto:email}{email} \\,|\\, \\href{url}{url}"}
```

2. Then output ONLY the LaTeX BODY content. Do NOT include \documentclass, \usepackage, \begin{document}, \end{document}, or \header.

LATEX COMMANDS TO USE:
- \section{Title}
- \role{Title | Company}{Dates}{Location}{Detail}  (first entry in section)
- \nextrole{Title | Company}{Dates}{Location}{Detail}  (subsequent entries)
- \begin{highlights} \item bullet text \end{highlights}
- \textbf{Category:} text\\[2pt]  (for Technical Skills)

ESCAPING RULES (CRITICAL - follow exactly):
- Dollar amounts: \$20M  (backslash before $)
- Ampersand in names: FP\&A, SG\&A  (backslash before &)
- Percent sign: 15\%  (backslash before %)
- Hash: \#
- Underscore in URLs is fine inside \href{}

CONTENT RULES:
- Keep ALL facts truthful. Never invent experience, companies, dates, or metrics.
- Never fabricate achievements — only rephrase real ones to better match job language.
- Preserve every single role, project, and section from the original resume.
"""


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
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
        )
        if response.status_code != 200:
            raise Exception(f"Claude API error ({response.status_code}): {response.text[:500]}")
        data = response.json()
        return data["content"][0]["text"]


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
    """Fix common LaTeX issues from AI-generated content."""
    # Remove preamble lines Claude might accidentally include
    body = re.sub(r"\\begin\{document\}", "", body)
    body = re.sub(r"\\end\{document\}", "", body)
    body = re.sub(r"\\documentclass.*\n?", "", body)
    body = re.sub(r"\\usepackage.*\n?", "", body)
    body = re.sub(r"^\\header\s*$", "", body, flags=re.MULTILINE)

    # Fix unescaped $ before digits (dollar amounts like $20M)
    body = re.sub(r'(?<!\\)\$(\d)', r'\\$\1', body)

    # Fix unescaped % (but not already escaped)
    body = re.sub(r'(?<!\\)%', r'\\%', body)

    # Fix unescaped & inside \item lines (company names like FP&A)
    lines = body.split("\n")
    fixed = []
    for line in lines:
        if line.strip().startswith("\\item"):
            # Escape & that isn't already escaped
            line = re.sub(r'(?<!\\)&', r'\\&', line)
        fixed.append(line)
    body = "\n".join(fixed)

    # Fix double escapes that might result
    body = body.replace("\\\\&", "\\&")
    body = body.replace("\\\\%", "\\%")
    body = re.sub(r'\\\\\$(\d)', r'\\$\1', body)

    return body


def build_full_latex(name: str, contact_line: str, body: str) -> str:
    full = LATEX_TEMPLATE.replace("<<NAME>>", name)
    full = full.replace("<<CONTACT_LINE>>", contact_line)
    full = full.replace("<<BODY>>", body)
    return full


@app.post("/api/tailor-json")
async def tailor_resume_json(
    resume_text: str = Form(...),
    job_description: str = Form(...),
):
    try:
        if not ANTHROPIC_API_KEY:
            return JSONResponse(content={
                "success": False, "pdf_base64": None, "latex": "",
                "error": "ANTHROPIC_API_KEY not configured. Add it in Railway Variables."
            })

        resume_text = process_resume_text(resume_text)

        user_message = f"""Here is the candidate's current resume:

---RESUME START---
{resume_text}
---RESUME END---

Here is the target job description:

---JOB DESCRIPTION START---
{job_description}
---JOB DESCRIPTION END---

Tailor this resume for the job. Output the JSON metadata block first, then the LaTeX body."""

        claude_response = await call_claude(SYSTEM_PROMPT, user_message)
        name, contact_line, latex_body = parse_claude_response(claude_response)
        latex_body = sanitize_latex(latex_body)
        full_latex = build_full_latex(name, contact_line, latex_body)

        # Try to compile
        pdf_base64_str = None
        compile_error = None
        try:
            pdf_path = compile_latex(full_latex)
            with open(pdf_path, "rb") as f:
                pdf_base64_str = base64.b64encode(f.read()).decode("utf-8")
        except Exception as first_err:
            # Auto-retry: ask Claude to fix the broken LaTeX
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
                pdf_path = compile_latex(full_latex)
                with open(pdf_path, "rb") as f:
                    pdf_base64_str = base64.b64encode(f.read()).decode("utf-8")
            except Exception as retry_err:
                compile_error = f"Compilation failed after auto-fix retry. Error: {str(first_err)[-500:]}"

        return JSONResponse(content={
            "success": pdf_base64_str is not None,
            "pdf_base64": pdf_base64_str,
            "latex": full_latex,
            "error": compile_error,
        })

    except ValueError as e:
        return JSONResponse(content={
            "success": False, "pdf_base64": None, "latex": "",
            "error": str(e)
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False, "pdf_base64": None, "latex": "",
            "error": f"Server error: {str(e)}"
        })


app.mount("/", StaticFiles(directory="static", html=True), name="static")
