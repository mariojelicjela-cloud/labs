from dataclasses import dataclass
from typing import List, Dict, Any
from pypdf import PdfReader

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]

class PDFFileLoader:
    def __init__(self, path: str, *, by_page: bool = True):
        self.path = path
        self.by_page = by_page

    def load(self) -> List[Document]:
        reader = PdfReader(self.path)
        docs: List[Document] = []

        if self.by_page:
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    continue

                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": self.path,
                            "page": i + 1,
                            "type": "pdf",
                        },
                    )
                )
        else:
            parts = []
            for page in reader.pages:
                t = (page.extract_text() or "").strip()
                if t:
                    parts.append(t)

            full_text = "\n\n".join(parts)
            if full_text.strip():
                docs.append(
                    Document(
                        text=full_text,
                        metadata={"source": self.path, "type": "pdf"},
                    )
                )

        return docs
