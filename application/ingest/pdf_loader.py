import fitz  # PyMuPDF

def load_pdf(file_path):
    """
    Extract the text from a PDF file page by page.
    """
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()
    return text