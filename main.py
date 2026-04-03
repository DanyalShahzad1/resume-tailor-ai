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


def _get_remaining_space_pt(latex_code: str, max_pages: int) -> float:
    r"""Measure remaining vertical space on the last target page in points.

    Inserts a \\write command at \\end{document} that records \\pagetotal
    (how much vertical content is on the current page) and \\pagegoal
    (total available height on the page).  After compilation we parse
    these values from the .log file.
    """
    probe = latex_code.replace(
        r"\end{document}",
        r"""
\makeatletter
\typeout{PAGEFIT::pagetotal=\the\pagetotal::pagegoal=\the\pagegoal}
\makeatother
\end{document}"""
    )
    tmpdir = tempfile.mkdtemp()
    tex_path = os.path.join(tmpdir, "probe.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(probe)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
        capture_output=True, text=True, timeout=30,
    )
    log_path = os.path.join(tmpdir, "probe.log")
    pagetotal = pagegoal = 0.0
    if os.path.exists(log_path):
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                m = re.search(r"PAGEFIT::pagetotal=([\d.]+)pt::pagegoal=([\d.]+)pt", line)
                if m:
                    pagetotal = float(m.group(1))
                    pagegoal = float(m.group(2))
    shutil.rmtree(tmpdir, ignore_errors=True)
    return pagegoal - pagetotal  # remaining space in pt


def _apply_spacing_multiplier(latex_code: str, mult: float) -> str:
    """Apply a spacing multiplier to all tunable spacing parameters.

    mult=1.0 → original values. mult>1 → expand spacing. mult<1 → shrink spacing.
    Values are clamped to sensible minimums.

    Escaping notes:
    - In regex PATTERN (raw string): r"\\\\" matches one literal backslash
      (but since we use r"", r"\\" is 2 chars which regex reads as: match 1 backslash)
    - In re.sub REPLACEMENT (f-string): "\\\\\\\\" produces one literal backslash
      (re.sub replacement processes \\\\ → \\)
    """
    def scale(base, minimum, maximum):
        return max(minimum, min(maximum, round(base * mult, 3)))

    m = latex_code

    # itemsep in highlights (no backslash prefix — simple match)
    new_itemsep = scale(0.6, 0.0, 4.0)
    m = re.sub(r"itemsep=[\d.]+pt", f"itemsep={new_itemsep}pt", m)

    # topsep in highlights
    new_topsep = scale(0.03, 0.005, 0.15)
    m = re.sub(r"topsep=[\d.]+cm", f"topsep={new_topsep}cm", m)

    # parsep in highlights
    new_parsep = scale(0.0, 0.0, 2.0)
    m = re.sub(r"parsep=[\d.]+pt", f"parsep={new_parsep}pt", m)

    # section title spacing: \titlespacing*{\section}{0pt}{X}{Y}
    new_sec_before = scale(0.13, 0.04, 0.40)
    new_sec_after = scale(0.07, 0.02, 0.20)
    m = re.sub(
        r"\\titlespacing\*\{\\section\}\{0pt\}\{[\d.]+cm\}\{[\d.]+cm\}",
        f"\\\\titlespacing*{{\\\\section}}{{0pt}}{{{new_sec_before}cm}}{{{new_sec_after}cm}}",
        m,
    )

    # \vspace between roles (\nextrole definition)
    new_role_gap = scale(0.04, 0.01, 0.18)
    m = re.sub(
        r"\\newcommand\{\\nextrole\}\[4\]\{\\vspace\{[\d.]+cm\}",
        f"\\\\newcommand{{\\\\nextrole}}[4]{{\\\\vspace{{{new_role_gap}cm}}",
        m,
    )

    # vspace after tabular* in baseRole definition
    new_tab_gap = scale(0.03, 0.005, 0.12)
    m = re.sub(
        r"\\end\{tabular\*\}\\vspace\{[\d.]+cm\}",
        f"\\\\end{{tabular*}}\\\\vspace{{{new_tab_gap}cm}}",
        m,
    )

    return m


def compile_latex_fit_pages(latex_code: str, max_pages: int = 1) -> str:
    """Compile LaTeX and auto-adjust spacing to fill exactly max_pages.

    Strategy:
    1. Compile and check page count.
    2. If content overflows (pages > max_pages): binary-search a spacing
       multiplier < 1.0 that shrinks spacing. If spacing alone isn't enough,
       progressively tighten margins, then reduce font size.
    3. If content is under (pages <= max_pages but space remains): binary-search
       a spacing multiplier > 1.0 that expands spacing to fill the page.
       After finding the best multiplier that stays on max_pages, use
       \vfill between sections to distribute any residual gap evenly.
    """
    pdf_path = compile_latex(latex_code)
    pages = get_pdf_page_count(pdf_path)

    # ── SHRINK: content overflows past max_pages ──────────────────────
    if pages > max_pages:
        best_path = pdf_path
        best_code = latex_code

        # Phase 1: binary-search spacing multiplier from 1.0 down to 0.1
        lo, hi = 0.1, 1.0
        for _ in range(8):
            mid = (lo + hi) / 2
            candidate = _apply_spacing_multiplier(latex_code, mid)
            tp = compile_latex(candidate)
            if get_pdf_page_count(tp) <= max_pages:
                # fits — try less aggressive shrink
                shutil.rmtree(os.path.dirname(best_path), ignore_errors=True)
                best_path = tp
                best_code = candidate
                lo = mid
            else:
                shutil.rmtree(os.path.dirname(tp), ignore_errors=True)
                hi = mid

        if get_pdf_page_count(best_path) <= max_pages:
            # Now expand within the remaining space (fall through to expand)
            latex_code = best_code
            shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
            pdf_path = best_path
            pages = get_pdf_page_count(pdf_path)
        else:
            # Phase 2: tighten margins
            m = best_code
            margin_steps = [
                (r"top=0.65cm,bottom=0.65cm", r"top=0.50cm,bottom=0.50cm"),
                (r"top=0.50cm,bottom=0.50cm", r"top=0.40cm,bottom=0.40cm"),
                (r"left=0.9cm,right=0.9cm", r"left=0.75cm,right=0.75cm"),
                (r"left=0.75cm,right=0.75cm", r"left=0.60cm,right=0.60cm"),
            ]
            for old, new in margin_steps:
                m = m.replace(old, new)
                tp = compile_latex(m)
                if get_pdf_page_count(tp) <= max_pages:
                    shutil.rmtree(os.path.dirname(best_path), ignore_errors=True)
                    best_path = tp
                    best_code = m
                    break
                shutil.rmtree(os.path.dirname(tp), ignore_errors=True)

            if get_pdf_page_count(best_path) > max_pages:
                # Phase 3: reduce font size
                for font_cmd in [r"\small", r"\footnotesize"]:
                    m = best_code.replace(r"\begin{document}", r"\begin{document}" + font_cmd)
                    tp = compile_latex(m)
                    if get_pdf_page_count(tp) <= max_pages:
                        shutil.rmtree(os.path.dirname(best_path), ignore_errors=True)
                        best_path = tp
                        best_code = m
                        break
                    shutil.rmtree(os.path.dirname(tp), ignore_errors=True)

            latex_code = best_code
            shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
            pdf_path = best_path
            pages = get_pdf_page_count(pdf_path)

    # ── EXPAND: content fits but doesn't fill the page ────────────────
    if pages <= max_pages:
        remaining = _get_remaining_space_pt(latex_code, max_pages)

        if remaining > 8.0:  # more than ~3mm of blank space at bottom
            # Binary-search spacing multiplier from 1.0 up
            best_path_expand = pdf_path
            best_code_expand = latex_code
            lo, hi = 1.0, 5.0

            for _ in range(10):
                mid = (lo + hi) / 2
                candidate = _apply_spacing_multiplier(latex_code, mid)
                tp = compile_latex(candidate)
                tp_pages = get_pdf_page_count(tp)
                if tp_pages <= max_pages:
                    shutil.rmtree(os.path.dirname(best_path_expand), ignore_errors=True)
                    best_path_expand = tp
                    best_code_expand = candidate
                    lo = mid
                else:
                    shutil.rmtree(os.path.dirname(tp), ignore_errors=True)
                    hi = mid

            latex_code = best_code_expand
            shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
            pdf_path = best_path_expand

            # Final pass: distribute residual space proportionally
            remaining = _get_remaining_space_pt(latex_code, max_pages)
            if remaining > 6.0:
                # Count section gaps (between sections) where we can add space
                section_positions = [m.start() for m in re.finditer(r"\\section\{", latex_code)]
                num_gaps = len(section_positions) - 1  # gaps between sections

                if num_gaps > 0:
                    # Calculate per-gap vspace to distribute
                    # Use 80% of remaining for between-section gaps, keep 20% for bottom
                    distribute = remaining * 0.85
                    per_gap_pt = distribute / num_gaps
                    bottom_pt = remaining - distribute

                    # Cap per-gap to avoid ugly huge gaps (max ~18pt = ~6mm)
                    max_gap_pt = 18.0
                    if per_gap_pt > max_gap_pt:
                        per_gap_pt = max_gap_pt
                        distribute = per_gap_pt * num_gaps
                        bottom_pt = remaining - distribute

                    # Insert calculated \vspace before each section except the first
                    parts = re.split(r"(\\section\{)", latex_code)
                    rebuilt = parts[0]
                    section_idx = 0
                    section_marker = "\\section{"
                    for i in range(1, len(parts)):
                        if parts[i] == section_marker:
                            if section_idx > 0:
                                rebuilt += f"\\vspace{{{per_gap_pt:.1f}pt}}\n"
                            section_idx += 1
                        rebuilt += parts[i]

                    # Add remaining space at bottom
                    if bottom_pt > 2.0:
                        rebuilt = rebuilt.replace(
                            "\\end{document}",
                            f"\\vspace{{\\fill}}\n\\end{{document}}"
                        )

                    tp = compile_latex(rebuilt)
                    if get_pdf_page_count(tp) <= max_pages:
                        shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
                        pdf_path = tp
                        latex_code = rebuilt

                        # If still significant space left, binary-search a larger per_gap
                        remaining2 = _get_remaining_space_pt(latex_code, max_pages)
                        if remaining2 > 15.0 and per_gap_pt < max_gap_pt:
                            # Try increasing per_gap further
                            lo2, hi2 = per_gap_pt, per_gap_pt + remaining2 / max(num_gaps, 1)
                            for _ in range(6):
                                mid2 = (lo2 + hi2) / 2
                                parts2 = re.split(r"(\\section\{)", best_code_expand)
                                rebuilt2 = parts2[0]
                                si2 = 0
                                for j in range(1, len(parts2)):
                                    if parts2[j] == section_marker:
                                        if si2 > 0:
                                            rebuilt2 += f"\\vspace{{{mid2:.1f}pt}}\n"
                                        si2 += 1
                                    rebuilt2 += parts2[j]
                                rebuilt2 = rebuilt2.replace(
                                    "\\end{document}",
                                    f"\\vspace{{\\fill}}\n\\end{{document}}"
                                )
                                tp2 = compile_latex(rebuilt2)
                                if get_pdf_page_count(tp2) <= max_pages:
                                    shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
                                    pdf_path = tp2
                                    latex_code = rebuilt2
                                    lo2 = mid2
                                else:
                                    shutil.rmtree(os.path.dirname(tp2), ignore_errors=True)
                                    hi2 = mid2
                    else:
                        shutil.rmtree(os.path.dirname(tp), ignore_errors=True)

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


# ── LaTeX → DOCX Conversion ─────────────────────────────────────────
def _unescape_latex(text: str) -> str:
    """Convert LaTeX escaped characters back to plain text."""
    text = text.replace(r"\$", "$")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\&", "&")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\~", "~")
    text = text.replace("~", " ")
    text = text.replace(r"\,", " ")
    text = text.replace(r"--", "–")
    text = text.replace(r"---", "—")
    # Remove remaining LaTeX commands we don't handle
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    # Handle \href{url}{text} → text
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    return text.strip()


def _parse_latex_body(latex_code: str):
    """Parse LaTeX resume into structured data for DOCX generation.

    Returns: (name, contact_text, sections)
    where sections is a list of:
      {"title": str, "entries": [{"role_line1": str, "role_line2": str,
        "date": str, "extra": str, "bullets": [str]}]}
    """
    # Extract name from pdftitle or \textbf in header
    name = "Candidate"
    name_m = re.search(r"pdftitle=\{(.+?)\s*--", latex_code)
    if name_m:
        name = name_m.group(1).strip()
    else:
        name_m = re.search(r"fontsize\{21pt\}.*?\\textbf\{(.+?)\}", latex_code)
        if name_m:
            name = name_m.group(1).strip()

    # Extract contact line (raw, between header center and \end{center})
    contact_text = ""
    contact_m = re.search(
        r"\\selectfont\s*\\textbf\{[^}]+\}\}\\\\.*?\n\s*(.+?)\n\s*\\end\{center\}",
        latex_code, re.DOTALL,
    )
    if contact_m:
        raw_contact = contact_m.group(1).strip()
        contact_text = _unescape_latex(raw_contact)
        contact_text = re.sub(r"\s*\|\s*", " | ", contact_text)
        contact_text = re.sub(r"\s+", " ", contact_text).strip()

    # Get the body (everything between \header and \end{document})
    body_m = re.search(r"\\header\s*\n(.*?)\\end\{document\}", latex_code, re.DOTALL)
    if not body_m:
        body_m = re.search(r"\\begin\{document\}.*?\\header\s*\n(.*?)\\end\{document\}", latex_code, re.DOTALL)
    body = body_m.group(1) if body_m else ""

    # Remove any \vspace{...} or \vfill inserted by the page-fitting algorithm
    body = re.sub(r"\\vfill\s*\n?", "", body)
    body = re.sub(r"\\vspace\{[^}]+\}\s*\n?", "", body)

    # Split into sections
    section_splits = re.split(r"\\section\{([^}]+)\}", body)
    # section_splits[0] is content before first section (usually empty)
    # then alternating: section_title, section_content, section_title, ...

    sections = []
    for i in range(1, len(section_splits), 2):
        sec_title = _unescape_latex(section_splits[i])
        sec_content = section_splits[i + 1] if i + 1 < len(section_splits) else ""

        entries = []

        # Find all \role and \nextrole commands
        role_pattern = r"\\(?:next)?role\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}"
        role_matches = list(re.finditer(role_pattern, sec_content))

        if role_matches:
            for j, rm in enumerate(role_matches):
                role_line1 = _unescape_latex(rm.group(1))  # Title | Company
                date = _unescape_latex(rm.group(2))
                role_line2 = _unescape_latex(rm.group(3))  # Location
                extra = _unescape_latex(rm.group(4))  # e.g. GPA, league

                # Get content between this role and the next (or section end)
                start = rm.end()
                end = role_matches[j + 1].start() if j + 1 < len(role_matches) else len(sec_content)
                role_content = sec_content[start:end]

                # Extract bullets
                bullets = []
                for item_m in re.finditer(r"\\item\s+(.+?)(?=\\item|\s*\\end\{highlights\}|$)",
                                          role_content, re.DOTALL):
                    bullet_text = _unescape_latex(item_m.group(1).strip())
                    bullet_text = re.sub(r"\s+", " ", bullet_text)
                    if bullet_text:
                        bullets.append(bullet_text)

                entries.append({
                    "role_line1": role_line1,
                    "date": date,
                    "role_line2": role_line2,
                    "extra": extra,
                    "bullets": bullets,
                })
        else:
            # No roles — probably Technical Skills or plain text section
            plain_text = sec_content.strip()
            # Parse \textbf{Category:} text\\[2pt] lines
            skill_lines = re.findall(
                r"\\textbf\{([^}]+)\}\s*(.+?)(?=\\textbf\{|\\\\|$)", plain_text, re.DOTALL
            )
            if skill_lines:
                for cat, content in skill_lines:
                    content = _unescape_latex(content.strip().rstrip("\\").strip())
                    entries.append({
                        "role_line1": "",
                        "date": "",
                        "role_line2": "",
                        "extra": "",
                        "bullets": [],
                        "skill_category": _unescape_latex(cat),
                        "skill_content": content,
                    })
            else:
                # Fallback: treat as plain text
                plain = _unescape_latex(plain_text)
                plain = re.sub(r"\\\\(\[[\d.]+pt\])?", "\n", plain)
                if plain.strip():
                    entries.append({
                        "role_line1": "", "date": "", "role_line2": "",
                        "extra": "", "bullets": [],
                        "plain_text": plain.strip(),
                    })

        sections.append({"title": sec_title, "entries": entries})

    return name, contact_text, sections


def latex_to_docx(latex_code: str) -> str:
    """Convert LaTeX resume to a DOCX file. Returns path to the generated .docx."""
    from docx import Document as DocxDocument
    from docx.shared import Pt, Inches, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
    from docx.oxml.ns import qn

    name, contact_text, sections = _parse_latex_body(latex_code)

    doc = DocxDocument()

    # ── Page setup: Letter, tight margins matching the LaTeX ──
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Cm(0.65)
    section.bottom_margin = Cm(0.65)
    section.left_margin = Cm(0.9)
    section.right_margin = Cm(0.9)

    # ── Style definitions ──
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(0)
    style.paragraph_format.line_spacing = Pt(12)

    # ── Name header ──
    name_para = doc.add_paragraph()
    name_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name_para.paragraph_format.space_after = Pt(2)
    name_run = name_para.add_run(name)
    name_run.bold = True
    name_run.font.size = Pt(20)
    name_run.font.name = "Calibri"

    # ── Contact line ──
    if contact_text:
        contact_para = doc.add_paragraph()
        contact_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        contact_para.paragraph_format.space_after = Pt(4)
        contact_run = contact_para.add_run(contact_text)
        contact_run.font.size = Pt(10)
        contact_run.font.name = "Calibri"

    # ── Sections ──
    for sec in sections:
        # Section heading with bottom border
        heading_para = doc.add_paragraph()
        heading_para.paragraph_format.space_before = Pt(6)
        heading_para.paragraph_format.space_after = Pt(3)
        heading_run = heading_para.add_run(sec["title"])
        heading_run.bold = True
        heading_run.font.size = Pt(12)
        heading_run.font.name = "Calibri"

        # Add bottom border to section heading
        pPr = heading_para._p.get_or_add_pPr()
        pBdr = pPr.makeelement(qn("w:pBdr"), {})
        bottom = pBdr.makeelement(qn("w:bottom"), {
            qn("w:val"): "single",
            qn("w:sz"): "6",
            qn("w:space"): "1",
            qn("w:color"): "000000",
        })
        pBdr.append(bottom)
        pPr.append(pBdr)

        for entry in sec["entries"]:
            # ── Skill lines (Technical Skills section) ──
            if entry.get("skill_category"):
                skill_para = doc.add_paragraph()
                skill_para.paragraph_format.space_before = Pt(1)
                skill_para.paragraph_format.space_after = Pt(1)
                cat_run = skill_para.add_run(entry["skill_category"] + " ")
                cat_run.bold = True
                cat_run.font.size = Pt(10)
                cat_run.font.name = "Calibri"
                content_run = skill_para.add_run(entry["skill_content"])
                content_run.font.size = Pt(10)
                content_run.font.name = "Calibri"
                continue

            # ── Plain text fallback ──
            if entry.get("plain_text"):
                plain_para = doc.add_paragraph()
                plain_para.paragraph_format.space_before = Pt(1)
                plain_run = plain_para.add_run(entry["plain_text"])
                plain_run.font.size = Pt(10)
                plain_run.font.name = "Calibri"
                continue

            # ── Role entry with tabbed layout ──
            if entry["role_line1"]:
                # Line 1: Role/Title (bold) ... Date (bold italic, right-aligned)
                role_para1 = doc.add_paragraph()
                role_para1.paragraph_format.space_before = Pt(3)
                role_para1.paragraph_format.space_after = Pt(0)

                # Add right tab stop at page content width
                content_width_twips = int((8.5 * 2.54 - 0.9 * 2) / 2.54 * 1440)
                tab_stops = role_para1.paragraph_format.tab_stops
                tab_stops.add_tab_stop(Pt(content_width_twips / 20),
                                       alignment=WD_TAB_ALIGNMENT.RIGHT)

                title_run = role_para1.add_run(entry["role_line1"])
                title_run.bold = True
                title_run.font.size = Pt(10)
                title_run.font.name = "Calibri"

                if entry["date"]:
                    role_para1.add_run("\t")
                    date_run = role_para1.add_run(entry["date"])
                    date_run.bold = True
                    date_run.italic = True
                    date_run.font.size = Pt(10)
                    date_run.font.name = "Calibri"

                # Line 2: Location (italic) ... Extra (italic, right-aligned)
                if entry["role_line2"] or entry["extra"]:
                    role_para2 = doc.add_paragraph()
                    role_para2.paragraph_format.space_before = Pt(0)
                    role_para2.paragraph_format.space_after = Pt(1)
                    tab_stops2 = role_para2.paragraph_format.tab_stops
                    tab_stops2.add_tab_stop(Pt(content_width_twips / 20),
                                            alignment=WD_TAB_ALIGNMENT.RIGHT)

                    if entry["role_line2"]:
                        loc_run = role_para2.add_run(entry["role_line2"])
                        loc_run.italic = True
                        loc_run.font.size = Pt(10)
                        loc_run.font.name = "Calibri"

                    if entry["extra"]:
                        role_para2.add_run("\t")
                        extra_run = role_para2.add_run(entry["extra"])
                        extra_run.italic = True
                        extra_run.font.size = Pt(10)
                        extra_run.font.name = "Calibri"

            # ── Bullet points ──
            for bullet_text in entry["bullets"]:
                bullet_para = doc.add_paragraph(style="List Bullet")
                bullet_para.paragraph_format.space_before = Pt(0.5)
                bullet_para.paragraph_format.space_after = Pt(0.5)
                bullet_para.paragraph_format.left_indent = Pt(11)
                bullet_para.paragraph_format.first_line_indent = Pt(-11)
                bullet_run = bullet_para.add_run(bullet_text)
                bullet_run.font.size = Pt(10)
                bullet_run.font.name = "Calibri"

    # Save to temp file
    tmpdir = tempfile.mkdtemp()
    docx_path = os.path.join(tmpdir, "tailored_resume.docx")
    doc.save(docx_path)
    return docx_path


@app.post("/api/download-docx")
async def download_docx(latex: str = Form(...)):
    """Convert the stored LaTeX to a DOCX and return as base64."""
    try:
        docx_path = latex_to_docx(latex)
        with open(docx_path, "rb") as f:
            docx_base64 = base64.b64encode(f.read()).decode("utf-8")
        shutil.rmtree(os.path.dirname(docx_path), ignore_errors=True)
        return JSONResponse(content={"success": True, "docx_base64": docx_base64})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": f"DOCX conversion failed: {str(e)}"})


app.mount("/", StaticFiles(directory="static", html=True), name="static")
