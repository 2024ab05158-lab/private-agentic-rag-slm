"""
pdf_loader.py
-------------------------------------------------

PDF Ingestion Module.

Supports:
1. Normal text based PDFs
2. Image / scanned PDFs using OCR

Used by:
- Core RAG Pipeline
- Agentic RAG Pipeline
"""


import fitz


from application.ocr.ocr_processor import (

    extract_text_using_ocr

)



def load_pdf(

        file_path

):


    """

    Extract text from PDF.

    First attempts normal PDF extraction.

    If extracted content is insufficient,
    automatically switches to OCR.

    """


    document = fitz.open(

        file_path

    )


    extracted_text = ""



    # ----------------------------------
    # Standard PDF Text Extraction
    # ----------------------------------


    for page in document:


        page_text = page.get_text()


        if page_text:


            extracted_text += (

                page_text

                +

                "\n"

            )



    document.close()



    # ----------------------------------
    # OCR Fallback
    # ----------------------------------


    if len(

        extracted_text.strip()

    ) < 100:


        print(

            "Low text detected. Running OCR..."

        )


        extracted_text = extract_text_using_ocr(

            file_path

        )


    else:


        print(

            "PDF text extracted successfully."

        )



    return extracted_text