from docx import Document
from PyPDF2 import PdfReader
from io import BytesIO

def parse_docx(file):
    file.seek(0)
    doc = Document(BytesIO(file.read()))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])

def parse_pdf(file):
    file.seek(0)
    reader = PdfReader(BytesIO(file.read()))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def parse_file(file):
    if file.name.endswith(".docx"):
        return parse_docx(file)
    elif file.name.endswith(".pdf"):
        return parse_pdf(file)
    else:
        return ""
