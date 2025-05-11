import PyPDF2
import docx

def read_txt(file_stream):
    return file_stream.read().decode('utf-8')

def read_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    return "\n".join(page.extract_text() for page in reader.pages)

def read_docx(file_stream):
    doc = docx.Document(file_stream)
    return "\n".join(p.text for p in doc.paragraphs)
