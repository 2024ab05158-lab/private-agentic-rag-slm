import json
import os


METADATA_FILE = "data/vector_db/index_metadata.json"


class IndexManager:

    def __init__(self):

        os.makedirs(
            "data/vector_db",
            exist_ok=True
        )

        if not os.path.exists(
            METADATA_FILE
        ):

            with open(
                METADATA_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    {
                        "documents": []
                    },
                    file,
                    indent=4
                )


    def load_metadata(self):

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)


    def save_metadata(
            self,
            data
    ):

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )


    def list_documents(self):

        return self.load_metadata()["documents"]


    def update_document(
            self,
            filename,
            last_modified,
            chunks
    ):

        metadata = self.load_metadata()

        docs = metadata["documents"]

        docs = [

            doc

            for doc in docs

            if doc["filename"] != filename

        ]

        docs.append(

            {

                "filename": filename,

                "indexed": True,

                "last_modified": last_modified,

                "chunks": chunks

            }

        )

        metadata["documents"] = docs

        self.save_metadata(
            metadata
        )


    def remove_document(
            self,
            filename
    ):

        metadata = self.load_metadata()

        metadata["documents"] = [

            doc

            for doc in metadata["documents"]

            if doc["filename"] != filename

        ]

        self.save_metadata(
            metadata
        )


    def clear_metadata(self):

        self.save_metadata(

            {

                "documents": []

            }

        )


    def rebuild_required(
            self,
            pdf_path
    ):

        filename = os.path.basename(
            pdf_path
        )

        modified = os.path.getmtime(
            pdf_path
        )

        for doc in self.list_documents():

            if (

                doc["filename"] == filename

                and

                doc["last_modified"] == modified

            ):

                return False

        return True