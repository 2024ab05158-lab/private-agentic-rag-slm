"""
ocr_processor.py
-------------------------------------------------

OCR Processing Module.

Responsibilities:

1. Extract images from PDF pages
2. Convert page images into text
3. Support scanned documents
4. Enhance RAG knowledge extraction

"""


import fitz

import pytesseract

from PIL import Image

import io



def extract_text_using_ocr(

        pdf_path

):


    """

    Extract text from PDF using OCR.

    Used when normal PDF text extraction
    does not return useful content.

    """


    extracted_text = ""


    document = fitz.open(

        pdf_path

    )


    for page_number, page in enumerate(

        document

    ):


        pix = page.get_pixmap(

            dpi=300

        )


        image = Image.open(

            io.BytesIO(

                pix.tobytes("png")

            )

        )


        page_text = pytesseract.image_to_string(

            image

        )


        if page_text.strip():


            extracted_text += (

                "\n\n"

                +

                f"OCR_PAGE_{page_number + 1}\n"

                +

                page_text

            )


    document.close()


    return extracted_text