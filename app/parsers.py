"""Robust Multi-Format Document Parsers for AMD Mini-Challenge 3 (RAG).

Supports:
- .pdf (datasheets, specs, handles encryption & withdrawn tags gracefully)
- .docx (word documents, roadmaps, paragraphs and tables)
- .xlsx (spreadsheets with multiple sheets, parts, pricing, lead times)
- .csv (exported ticket/bug databases)
- .txt / .log (production logs, release notes)
- .py (source code constants, default values)
- .png / .jpg / .jpeg (multimodal images via OCR)

Defensive design:
- Never raises on unreadable files, chmod 000, unknown formats, or encrypted PDFs.
- Returns structured DocumentChunk objects with normalized relative paths.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mc3_parsers")


@dataclass
class DocumentChunk:
    rel_path: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    is_superseded: bool = False
    is_encrypted: bool = False


class RobustDocumentParser:
    def __init__(self, ocr_fn=None):
        self.ocr_fn = ocr_fn

    def parse_file(self, full_path: Path, rel_path: str) -> List[DocumentChunk]:
        """Safely parse a single file into DocumentChunks. Never raises."""
        try:
            # Check file access permissions (handles chmod 000)
            if not os.path.exists(full_path):
                return []
            if not os.access(full_path, os.R_OK):
                logger.warning(f"No read permission for {rel_path}, skipping.")
                return []

            ext = full_path.suffix.lower()
            if ext == ".pdf":
                return self._parse_pdf(full_path, rel_path)
            elif ext == ".docx":
                return self._parse_docx(full_path, rel_path)
            elif ext == ".xlsx":
                return self._parse_xlsx(full_path, rel_path)
            elif ext == ".csv":
                return self._parse_csv(full_path, rel_path)
            elif ext in (".txt", ".log"):
                return self._parse_text(full_path, rel_path)
            elif ext == ".py":
                return self._parse_python(full_path, rel_path)
            elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                return self._parse_image(full_path, rel_path)
            else:
                logger.info(f"Skipping unknown or unsupported file extension {ext}: {rel_path}")
                return []
        except Exception as e:
            logger.warning(f"Error parsing {rel_path}: {e}")
            return []

    def _parse_pdf(self, path: Path, rel_path: str) -> List[DocumentChunk]:
        chunks = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            if reader.is_encrypted:
                # Encrypted PDF trap: mark as encrypted and return empty
                logger.info(f"File {rel_path} is password-encrypted, cannot extract.")
                return [DocumentChunk(rel_path=rel_path, chunk_id=f"{rel_path}:enc", content="", metadata={"encrypted": True}, is_encrypted=True)]

            is_superseded = False
            filename_lower = path.name.lower()
            if "withdrawn" in filename_lower or "obsolete" in filename_lower or "deprecated" in filename_lower:
                is_superseded = True

            full_text = []
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    if text.strip():
                        if "WITHDRAWN" in text or "SUPERSEDED" in text:
                            is_superseded = True
                        full_text.append(text.strip())
                except Exception as e:
                    logger.warning(f"Page extract error in {rel_path} p{page_num}: {e}")

            joined_text = "\n\n".join(full_text)
            if joined_text.strip():
                chunks.append(DocumentChunk(
                    rel_path=rel_path,
                    chunk_id=f"{rel_path}:full",
                    content=joined_text,
                    metadata={"pages": len(reader.pages), "type": "pdf"},
                    is_superseded=is_superseded,
                ))
        except Exception as e:
            logger.warning(f"Failed to parse PDF {rel_path}: {e}")
        return chunks

    def _parse_docx(self, path: Path, rel_path: str) -> List[DocumentChunk]:
        chunks = []
        try:
            import docx
            doc = docx.Document(str(path))
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            # Extract tables
            table_texts = []
            for t_idx, table in enumerate(doc.tables):
                rows = []
                for row in table.rows:
                    cell_vals = [c.text.strip() for c in row.cells]
                    rows.append(" | ".join(cell_vals))
                if rows:
                    table_texts.append(f"[Table {t_idx+1}]\n" + "\n".join(rows))

            combined = "\n\n".join(paras + table_texts)
            if combined.strip():
                chunks.append(DocumentChunk(
                    rel_path=rel_path,
                    chunk_id=f"{rel_path}:full",
                    content=combined,
                    metadata={"paragraphs": len(paras), "tables": len(doc.tables), "type": "docx"},
                ))
        except Exception as e:
            logger.warning(f"Failed to parse DOCX {rel_path}: {e}")
        return chunks

    def _parse_xlsx(self, path: Path, rel_path: str) -> List[DocumentChunk]:
        chunks = []
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(path), data_only=True)
            sheet_chunks = []
            for sheetname in wb.sheetnames:
                sheet = wb[sheetname]
                rows = list(sheet.iter_rows(values_only=True))
                if not rows:
                    continue
                
                header = [str(c or "").strip() for c in rows[0]]
                records = []
                for r_idx, row in enumerate(rows[1:], start=2):
                    if not any(row):
                        continue
                    row_dict = {}
                    row_strs = []
                    for h_idx, cell in enumerate(row):
                        h_name = header[h_idx] if h_idx < len(header) and header[h_idx] else f"col_{h_idx+1}"
                        val_str = str(cell).strip() if cell is not None else ""
                        row_dict[h_name] = val_str
                        row_strs.append(f"{h_name}={val_str}")
                    records.append(f"Row {r_idx}: {', '.join(row_strs)}")

                content = f"Sheet: {sheetname}\n" + "\n".join(records)
                sheet_chunks.append(content)

            if sheet_chunks:
                chunks.append(DocumentChunk(
                    rel_path=rel_path,
                    chunk_id=f"{rel_path}:full",
                    content="\n\n".join(sheet_chunks),
                    metadata={"sheets": wb.sheetnames, "type": "xlsx"},
                ))
        except Exception as e:
            logger.warning(f"Failed to parse XLSX {rel_path}: {e}")
        return chunks

    def _parse_csv(self, path: Path, rel_path: str) -> List[DocumentChunk]:
        chunks = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if not rows:
                    return []
                
                header = [h.strip() for h in rows[0]]
                records = []
                for r_idx, row in enumerate(rows[1:], start=2):
                    if not any(row):
                        continue
                    row_strs = []
                    for h_idx, val in enumerate(row):
                        h_name = header[h_idx] if h_idx < len(header) else f"col_{h_idx+1}"
                        row_strs.append(f"{h_name}={val.strip()}")
                    records.append(f"Row {r_idx}: {', '.join(row_strs)}")

                content = f"Columns: {', '.join(header)}\n" + "\n".join(records)
                chunks.append(DocumentChunk(
                    rel_path=rel_path,
                    chunk_id=f"{rel_path}:full",
                    content=content,
                    metadata={"rows": len(records), "type": "csv"},
                ))
        except Exception as e:
            logger.warning(f"Failed to parse CSV {rel_path}: {e}")
        return chunks

    def _parse_text(self, path: Path, rel_path: str) -> List[DocumentChunk]:
        chunks = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    chunks.append(DocumentChunk(
                        rel_path=rel_path,
                        chunk_id=f"{rel_path}:full",
                        content=content,
                        metadata={"type": "text", "lines": len(content.splitlines())},
                    ))
        except Exception as e:
            logger.warning(f"Failed to parse text file {rel_path}: {e}")
        return chunks

    def _parse_python(self, path: Path, rel_path: str) -> List[DocumentChunk]:
        chunks = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    # Also extract constants like CONST = value
                    constants = []
                    for line in content.splitlines():
                        m = re.match(r"^([A-Z0-9_]+)\s*=\s*(.+)$", line.strip())
                        if m:
                            constants.append(f"{m.group(1)} = {m.group(2)}")
                    
                    extra_meta = {"constants": constants, "type": "python"}
                    chunks.append(DocumentChunk(
                        rel_path=rel_path,
                        chunk_id=f"{rel_path}:full",
                        content=content,
                        metadata=extra_meta,
                    ))
        except Exception as e:
            logger.warning(f"Failed to parse python file {rel_path}: {e}")
        return chunks

    def _parse_image(self, path: Path, rel_path: str) -> List[DocumentChunk]:
        chunks = []
        try:
            ocr_text = ""
            if self.ocr_fn:
                ocr_text = self.ocr_fn(str(path))
            else:
                # Built-in fallback OCR attempt
                ocr_text = self._builtin_ocr(path)

            if ocr_text and ocr_text.strip():
                content = f"[IMAGE CONTENT: {rel_path}]\n{ocr_text.strip()}"
                chunks.append(DocumentChunk(
                    rel_path=rel_path,
                    chunk_id=f"{rel_path}:ocr",
                    content=content,
                    metadata={"type": "image"},
                ))
            else:
                logger.info(f"No OCR text extracted from {rel_path}")
        except Exception as e:
            logger.warning(f"Failed to OCR image {rel_path}: {e}")
        return chunks

    def _builtin_ocr(self, path: Path) -> str:
        """Attempt rapidocr with spatial 2D alignment, fallback to easyocr or pytesseract."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            res, _ = engine(str(path))
            if res:
                # Group items by 2D coordinates for diagram/pinout table extraction
                pins = []
                signals = []
                other_lines = []
                for box, txt, score in res:
                    txt = txt.strip()
                    x_center = (box[0][0] + box[2][0]) / 2.0
                    y_center = (box[0][1] + box[2][1]) / 2.0
                    # Check if pin identifier like B11, B12, A1, etc.
                    if re.match(r"^[A-Z]\d{1,2}$", txt):
                        pins.append((x_center, y_center, txt))
                    else:
                        signals.append((x_center, y_center, txt))

                paired_lines = []
                if pins and signals:
                    # Match each pin with the signal closest in x-coordinate
                    for px, py, pin_label in sorted(pins, key=lambda p: p[0]):
                        # Find signal with similar x
                        closest_sig = min(signals, key=lambda s: abs(s[0] - px) + abs(s[1] - py) * 0.1)
                        if abs(closest_sig[0] - px) < 40:
                            paired_lines.append(f"Pin {pin_label}: {closest_sig[2]}")

                all_text = " ".join([line[1] for line in res])
                if paired_lines:
                    return all_text + "\n[Pinout Mapping]\n" + "\n".join(paired_lines)
                return all_text
        except Exception:
            pass

        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            res = reader.readtext(str(path), detail=0)
            return " ".join(res)
        except Exception:
            pass

        try:
            import pytesseract
            from PIL import Image
            img = Image.open(str(path))
            return pytesseract.image_to_string(img)
        except Exception:
            pass

        return ""
