from application.ocr.ocr_processor import extract_text_using_ocr


pdf_path = "data/documents/23H2 Version Upgrade Guide.pdf"


text = extract_text_using_ocr(
    pdf_path
)


print(
    text[:1000]
)