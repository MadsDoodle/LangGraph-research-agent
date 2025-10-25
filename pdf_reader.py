from langchain_core.tools import tool
import io
import PyPDF2
import requests
from pathlib import Path
from typing import Union, Dict, Optional
import re
from datetime import datetime
from tqdm import tqdm
import logging

#for OCR support 
try:
    from pdf2image import convert_from_bytes, convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR libraries not available. Install with: pip install pdf2image pytesseract pillow")

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text from PDF
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove weird characters but keep common punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\'\"\n]', '', text)
    
    # Fix common OCR errors
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_metadata(pdf_reader: PyPDF2.PdfReader) -> Dict:
    """Extract metadata from PDF
    
    Args:
        pdf_reader: PyPDF2 PdfReader object
        
    Returns:
        Dictionary containing PDF metadata
    """
    metadata = {
        "num_pages": len(pdf_reader.pages),
        "title": None,
        "author": None,
        "subject": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "modification_date": None
    }
    
    try:
        if pdf_reader.metadata:
            metadata["title"] = pdf_reader.metadata.get("/Title", None)
            metadata["author"] = pdf_reader.metadata.get("/Author", None)
            metadata["subject"] = pdf_reader.metadata.get("/Subject", None)
            metadata["creator"] = pdf_reader.metadata.get("/Creator", None)
            metadata["producer"] = pdf_reader.metadata.get("/Producer", None)
            
            # Parse dates if available
            creation_date = pdf_reader.metadata.get("/CreationDate", None)
            if creation_date:
                metadata["creation_date"] = creation_date
                
            mod_date = pdf_reader.metadata.get("/ModDate", None)
            if mod_date:
                metadata["modification_date"] = mod_date
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")
    
    return metadata


def extract_text_with_ocr(pdf_source: Union[str, bytes], is_url: bool = True) -> str:
    """Extract text from PDF using OCR (for scanned PDFs)
    
    Args:
        pdf_source: Either URL string or bytes content of PDF
        is_url: Whether pdf_source is a URL (True) or local path (False)
        
    Returns:
        Extracted text using OCR
    """
    if not OCR_AVAILABLE:
        raise ImportError("OCR libraries not installed. Install with: pip install pdf2image pytesseract pillow")
    
    logger.info("[OCR] Converting PDF to images...")
    
    try:
        if is_url:
            # Download PDF first
            response = requests.get(pdf_source, timeout=60)
            images = convert_from_bytes(response.content)
        else:
            # Local file
            images = convert_from_path(pdf_source)
        
        logger.info(f"[OCR] Processing {len(images)} pages with OCR...")
        
        text = ""
        for i, image in enumerate(tqdm(images, desc="OCR Processing"), 1):
            page_text = pytesseract.image_to_string(image)
            text += f"\n--- Page {i} ---\n{page_text}\n"
        
        logger.info(f"[OCR] Extracted {len(text)} characters using OCR")
        return text
        
    except Exception as e:
        logger.error(f"[OCR] Error during OCR processing: {e}")
        raise

import re
from typing import Dict, List, Optional
import PyPDF2

def extract_title_from_text(text: str, max_lines: int = 10) -> Optional[str]:
    """Extract title from the first few lines of PDF text
    
    Args:
        text: Extracted text from PDF
        max_lines: Number of lines to check for title
        
    Returns:
        Extracted title or None
    """
    if not text:
        return None
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return None
    
    # Strategy 1: Look for title-like patterns in first few lines
    for i, line in enumerate(lines[:max_lines]):
        # Skip very short lines (likely not titles)
        if len(line) < 10:
            continue
        
        # Skip lines that look like headers/footers
        if re.match(r'^\d+$', line):  # Just a number
            continue
        if re.match(r'^page\s+\d+', line, re.IGNORECASE):
            continue
        if '@' in line or 'http' in line.lower():  # Email or URL
            continue
        
        # Title characteristics:
        # - Usually appears early
        # - Often in title case or all caps
        # - Not too long (usually < 200 chars)
        # - Doesn't end with common sentence enders followed by lowercase
        
        if len(line) < 200:
            # Check if it looks like a title
            words = line.split()
            if len(words) >= 3:  # At least 3 words
                # Check for title case or all caps
                capitalized_words = sum(1 for w in words if w[0].isupper() or w.isupper())
                if capitalized_words / len(words) > 0.5:  # More than 50% capitalized
                    return line
    
    # Strategy 2: If no clear title found, return first substantial line
    for line in lines[:5]:
        if len(line) > 15 and len(line) < 200:
            return line
    
    return None


def extract_authors_from_text(text: str, max_lines: int = 30) -> List[str]:
    """Extract authors from PDF text
    
    Args:
        text: Extracted text from PDF
        max_lines: Number of lines to check
        
    Returns:
        List of author names
    """
    authors = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Common author patterns
    author_patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$',  # John Doe
        r'^([A-Z]\.\s*[A-Z][a-z]+)$',  # J. Doe
        r'^([A-Z][a-z]+(?:\s+[A-Z]\.)+\s*[A-Z][a-z]+)$',  # John A. Doe
    ]
    
    # Keywords that might precede author list
    author_keywords = ['author', 'by', 'written by']
    
    for i, line in enumerate(lines[:max_lines]):
        # Check if line indicates authors
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in author_keywords):
            # Check next few lines for names
            for next_line in lines[i+1:i+5]:
                for pattern in author_patterns:
                    if re.match(pattern, next_line):
                        authors.append(next_line)
        
        # Also check lines that match author patterns directly
        for pattern in author_patterns:
            if re.match(pattern, line) and line not in authors:
                # Verify it's not a title or other content
                if len(line.split()) <= 4:  # Names usually 2-4 words
                    authors.append(line)
    
    return authors[:10]  # Limit to reasonable number


def extract_abstract(text: str, max_chars: int = 2000) -> Optional[str]:
    """Extract abstract from PDF text
    
    Args:
        text: Extracted text from PDF
        max_chars: Maximum characters to return
        
    Returns:
        Abstract text or None
    """
    # Look for abstract section
    abstract_pattern = r'(?i)abstract\s*[:.]?\s*(.*?)(?=\n\s*\n|\n\s*(?:introduction|keywords|1\.|I\.))'
    
    match = re.search(abstract_pattern, text, re.DOTALL)
    if match:
        abstract = match.group(1).strip()
        # Clean up
        abstract = re.sub(r'\s+', ' ', abstract)
        return abstract[:max_chars]
    
    return None


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from PDF text
    
    Args:
        text: Extracted text from PDF
        
    Returns:
        List of keywords
    """
    # Look for keywords section
    keywords_pattern = r'(?i)keywords?\s*[:.]?\s*(.*?)(?=\n\s*\n)'
    
    match = re.search(keywords_pattern, text[:5000])  # Check first 5000 chars
    if match:
        keywords_text = match.group(1).strip()
        # Split by common delimiters
        keywords = re.split(r'[,;·•]', keywords_text)
        keywords = [k.strip() for k in keywords if k.strip()]
        return keywords[:10]  # Limit to 10 keywords
    
    return []


def extract_doi(text: str) -> Optional[str]:
    """Extract DOI from PDF text
    
    Args:
        text: Extracted text from PDF
        
    Returns:
        DOI string or None
    """
    # DOI pattern
    doi_pattern = r'\b(10\.\d{4,}(?:\.\d+)*\/(?:(?!["&\'<>])\S)+)\b'
    
    match = re.search(doi_pattern, text[:3000])  # Check first 3000 chars
    if match:
        return match.group(1)
    
    return None


def extract_arxiv_id(text: str) -> Optional[str]:
    """Extract arXiv ID from PDF text
    
    Args:
        text: Extracted text from PDF
        
    Returns:
        arXiv ID or None
    """
    # arXiv ID patterns
    patterns = [
        r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:3000], re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def extract_publication_info(text: str) -> Dict[str, Optional[str]]:
    """Extract publication venue, conference, journal info
    
    Args:
        text: Extracted text from PDF
        
    Returns:
        Dictionary with publication info
    """
    info = {
        "venue": None,
        "conference": None,
        "journal": None,
        "year": None
    }
    
    # Check first 2000 characters for publication info
    header = text[:2000]
    
    # Look for conference patterns
    conf_patterns = [
        r'(?i)((?:conference|workshop|symposium|proceedings)[^\n]{0,100})',
        r'(?i)((?:ICML|NeurIPS|ICLR|CVPR|ICCV|ECCV|ACL|EMNLP|AAAI|IJCAI)[^\n]{0,50})'
    ]
    
    for pattern in conf_patterns:
        match = re.search(pattern, header)
        if match:
            info["conference"] = match.group(1).strip()[:100]
            break
    
    # Look for journal patterns
    journal_pattern = r'(?i)((?:journal|transactions)[^\n]{0,100})'
    match = re.search(journal_pattern, header)
    if match:
        info["journal"] = match.group(1).strip()[:100]
    
    # Extract year
    year_pattern = r'\b(19|20)\d{2}\b'
    years = re.findall(year_pattern, header)
    if years:
        # Get the most recent reasonable year
        years = [int(y) for y in years if 1990 <= int(y) <= 2030]
        if years:
            info["year"] = str(max(years))
    
    # Set venue (prefer conference, then journal)
    info["venue"] = info["conference"] or info["journal"]
    
    return info


def extract_enhanced_metadata(pdf_reader: PyPDF2.PdfReader, text: str = "") -> Dict:
    """Extract comprehensive metadata from PDF
    
    Args:
        pdf_reader: PyPDF2 PdfReader object
        text: Extracted text from PDF (optional, for text-based extraction)
        
    Returns:
        Dictionary containing comprehensive PDF metadata
    """
    logger.info("[Metadata] Starting enhanced metadata extraction")
    
    metadata = {
        # Basic PDF metadata
        "num_pages": len(pdf_reader.pages),
        "pdf_title": None,
        "pdf_author": None,
        "pdf_subject": None,
        "pdf_creator": None,
        "pdf_producer": None,
        "pdf_creation_date": None,
        "pdf_modification_date": None,
        
        # Extracted from content
        "title": None,
        "authors": [],
        "abstract": None,
        "keywords": [],
        "doi": None,
        "arxiv_id": None,
        
        # Publication info
        "venue": None,
        "conference": None,
        "journal": None,
        "year": None,
        
        # Statistics
        "char_count": len(text) if text else 0,
        "word_count": len(text.split()) if text else 0,
    }
    
    # Extract PDF metadata
    try:
        if pdf_reader.metadata:
            metadata["pdf_title"] = pdf_reader.metadata.get("/Title", None)
            metadata["pdf_author"] = pdf_reader.metadata.get("/Author", None)
            metadata["pdf_subject"] = pdf_reader.metadata.get("/Subject", None)
            metadata["pdf_creator"] = pdf_reader.metadata.get("/Creator", None)
            metadata["pdf_producer"] = pdf_reader.metadata.get("/Producer", None)
            
            creation_date = pdf_reader.metadata.get("/CreationDate", None)
            if creation_date:
                metadata["pdf_creation_date"] = creation_date
                
            mod_date = pdf_reader.metadata.get("/ModDate", None)
            if mod_date:
                metadata["pdf_modification_date"] = mod_date
    except Exception as e:
        logger.warning(f"[Metadata] Error extracting PDF metadata: {e}")
    
    # Extract from text content if available
    if text:
        logger.info("[Metadata] Extracting metadata from text content")
        
        # Title extraction (prefer content title over PDF metadata title)
        content_title = extract_title_from_text(text)
        metadata["title"] = content_title or metadata["pdf_title"]
        
        # Authors
        metadata["authors"] = extract_authors_from_text(text)
        
        # If no authors found in text, use PDF author
        if not metadata["authors"] and metadata["pdf_author"]:
            metadata["authors"] = [metadata["pdf_author"]]
        
        # Abstract
        metadata["abstract"] = extract_abstract(text)
        
        # Keywords
        metadata["keywords"] = extract_keywords(text)
        
        # DOI and arXiv ID
        metadata["doi"] = extract_doi(text)
        metadata["arxiv_id"] = extract_arxiv_id(text)
        
        # Publication info
        pub_info = extract_publication_info(text)
        metadata["venue"] = pub_info["venue"]
        metadata["conference"] = pub_info["conference"]
        metadata["journal"] = pub_info["journal"]
        metadata["year"] = pub_info["year"]
    
    logger.info(f"[Metadata] Extracted title: {metadata['title']}")
    logger.info(f"[Metadata] Found {len(metadata['authors'])} authors")
    logger.info(f"[Metadata] Found {len(metadata['keywords'])} keywords")
    
    return metadata


def format_metadata_display(metadata: Dict) -> str:
    """Format metadata for nice display
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PDF METADATA")
    lines.append("=" * 80)
    
    # Title
    if metadata.get("title"):
        lines.append(f"\nTitle: {metadata['title']}")
    
    # Authors
    if metadata.get("authors"):
        lines.append(f"\nAuthors:")
        for i, author in enumerate(metadata["authors"], 1):
            lines.append(f"  {i}. {author}")
    
    # Publication Info
    if metadata.get("venue") or metadata.get("year"):
        lines.append(f"\nPublication:")
        if metadata.get("venue"):
            lines.append(f"  Venue: {metadata['venue']}")
        if metadata.get("year"):
            lines.append(f"  Year: {metadata['year']}")
    
    # Identifiers
    if metadata.get("doi") or metadata.get("arxiv_id"):
        lines.append(f"\nIdentifiers:")
        if metadata.get("doi"):
            lines.append(f"  DOI: {metadata['doi']}")
        if metadata.get("arxiv_id"):
            lines.append(f"  arXiv: {metadata['arxiv_id']}")
    
    # Abstract
    if metadata.get("abstract"):
        lines.append(f"\nAbstract:")
        abstract = metadata["abstract"]
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        lines.append(f"  {abstract}")
    
    # Keywords
    if metadata.get("keywords"):
        lines.append(f"\nKeywords: {', '.join(metadata['keywords'])}")
    
    # Statistics
    lines.append(f"\nStatistics:")
    lines.append(f"  Pages: {metadata.get('num_pages', 'N/A')}")
    lines.append(f"  Characters: {metadata.get('char_count', 'N/A'):,}")
    lines.append(f"  Words: {metadata.get('word_count', 'N/A'):,}")
    
    # PDF Metadata
    if metadata.get("pdf_creator") or metadata.get("pdf_producer"):
        lines.append(f"\nPDF Information:")
        if metadata.get("pdf_creator"):
            lines.append(f"  Creator: {metadata['pdf_creator']}")
        if metadata.get("pdf_producer"):
            lines.append(f"  Producer: {metadata['pdf_producer']}")
        if metadata.get("pdf_creation_date"):
            lines.append(f"  Created: {metadata['pdf_creation_date']}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

def read_pdf_advanced(
    source: str,
    use_ocr: bool = False,
    auto_ocr_fallback: bool = True,
    clean: bool = True,
    extract_meta: bool = True,
    show_progress: bool = True
) -> Dict[str, Union[str, Dict, int]]:
    """Advanced PDF reading with multiple features
    
    Args:
        source: URL or local file path to PDF
        use_ocr: Force use of OCR for text extraction
        auto_ocr_fallback: Automatically try OCR if regular extraction fails
        clean: Apply text cleaning
        extract_meta: Extract PDF metadata
        show_progress: Show progress bar during extraction
        
    Returns:
        Dictionary containing:
            - text: Extracted text content
            - metadata: PDF metadata (if extract_meta=True)
            - num_pages: Number of pages
            - char_count: Number of characters extracted
            - extraction_method: Method used (pypdf2 or ocr)
    """
    logger.info(f"[PDF Reader] Starting to read PDF from: {source[:100]}...")
    
    is_url = source.startswith(('http://', 'https://'))
    
    # Force OCR if requested
    if use_ocr:
        logger.info("[PDF Reader] Using OCR mode (forced)")
        text = extract_text_with_ocr(source, is_url=is_url)
        if clean:
            text = clean_text(text)
        return {
            "text": text,
            "metadata": {},
            "num_pages": text.count("--- Page"),
            "char_count": len(text),
            "extraction_method": "ocr"
        }
    
    # Try regular extraction first
    try:
        # Load PDF
        if is_url:
            logger.info("[PDF Reader] Downloading PDF from URL...")
            response = requests.get(source, timeout=60)
            pdf_file = io.BytesIO(response.content)
        else:
            logger.info("[PDF Reader] Reading local PDF file...")
            if not Path(source).exists():
                raise FileNotFoundError(f"PDF file not found: {source}")
            pdf_file = open(source, 'rb')
        
        # Read PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        
        logger.info(f"[PDF Reader] PDF has {num_pages} pages")
        
        # Extract text first (needed for metadata extraction)
        text = ""
        pages_iterator = enumerate(pdf_reader.pages, 1)
        
        if show_progress:
            pages_iterator = tqdm(pages_iterator, total=num_pages, desc="Extracting text")
        
        for i, page in pages_iterator:
            if not show_progress:
                logger.debug(f"[PDF Reader] Extracting page {i}/{num_pages}")
            page_text = page.extract_text()
            text += page_text + "\n"
        
        
        
        # Check if extraction was successful
        extraction_method = "pypdf2"
        if len(text.strip()) < 100 and auto_ocr_fallback and OCR_AVAILABLE:
            logger.warning("[PDF Reader] Very little text extracted. Falling back to OCR...")
            text = extract_text_with_ocr(source, is_url=is_url)
            extraction_method = "ocr_fallback"
        
        # Clean text if requested (before metadata extraction to get better results)
        if clean:
            logger.info("[PDF Reader] Cleaning extracted text...")
            text = clean_text(text)
        
        # Extract metadata with the extracted text
        metadata = {}
        if extract_meta:
            logger.info("[PDF Reader] Extracting enhanced metadata...")
            metadata = extract_enhanced_metadata(pdf_reader, text)

        # Close file if local
        if not is_url:
            pdf_file.close()
        
        logger.info(f"[PDF Reader] Successfully extracted {len(text)} characters")
        
        return {
            "text": text,
            "metadata": metadata,
            "num_pages": num_pages,
            "char_count": len(text),
            "extraction_method": extraction_method
        }
        
    except Exception as e:
        logger.error(f"[PDF Reader] Error during extraction: {e}")
        
        # Try OCR as fallback
        if auto_ocr_fallback and OCR_AVAILABLE and not use_ocr:
            logger.info("[PDF Reader] Attempting OCR as fallback...")
            try:
                text = extract_text_with_ocr(source, is_url=is_url)
                if clean:
                    text = clean_text(text)
                return {
                    "text": text,
                    "metadata": {},
                    "num_pages": text.count("--- Page"),
                    "char_count": len(text),
                    "extraction_method": "ocr_fallback"
                }
            except Exception as ocr_error:
                logger.error(f"[PDF Reader] OCR fallback also failed: {ocr_error}")
        
        raise


@tool
def read_pdf(source: str, use_ocr: bool = False, clean: bool = True) -> str:
    """Read and extract text from a PDF file from URL or local path.
    
    Args:
        source: URL or local file path to the PDF
        use_ocr: Force use of OCR for scanned PDFs (default: False, auto-detects)
        clean: Apply text cleaning and normalization (default: True)
        
    Returns:
        The extracted text content from the PDF
    """
    try:
        result = read_pdf_advanced(
            source=source,
            use_ocr=use_ocr,
            auto_ocr_fallback=True,
            clean=clean,
            extract_meta=False,
            show_progress=True
        )
        return result["text"]
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise


@tool
def read_pdf_with_metadata(source: str, use_ocr: bool = False, clean: bool = True) -> Dict:
    """Read PDF and return both text and metadata.
    
    Args:
        source: URL or local file path to the PDF
        use_ocr: Force use of OCR for scanned PDFs (default: False, auto-detects)
        clean: Apply text cleaning and normalization (default: True)
        
    Returns:
        Dictionary with 'text', 'metadata', 'num_pages', 'char_count', 'extraction_method'
    """
    try:
        result = read_pdf_advanced(
            source=source,
            use_ocr=use_ocr,
            auto_ocr_fallback=True,
            clean=clean,
            extract_meta=True,
            show_progress=True
        )
        return result
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Test with URL
    print("=" * 80)
    print("Testing PDF reading from URL")
    print("=" * 80)
    
    url = "https://arxiv.org/pdf/1706.03762.pdf"  # Attention is All You Need paper
    
    try:
        # Simple text extraction
        text = read_pdf.invoke({"source": url, "clean": True})
        print(f"\n✓ Extracted {len(text)} characters")
        print(f"First 500 characters:\n{text[:500]}...\n")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test with metadata
    print("=" * 80)
    print("Testing PDF reading with metadata")
    print("=" * 80)
    
    try:
        result = read_pdf_with_metadata.invoke({"source": url, "clean": True})
        print(f"\n✓ Extraction successful!")
        print(f"  Pages: {result['num_pages']}")
        print(f"  Characters: {result['char_count']}")
        print(f"  Method: {result['extraction_method']}")
        print(f"\nMetadata:")
        for key, value in result['metadata'].items():
            if value:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test with local file (if you have one)
    print("\n" + "=" * 80)
    print("Testing PDF reading from local file")
    print("=" * 80)
    
    local_path = "research_papers/arxiv/LayerComposer Interactive Personalized T2I via Spatially-Aware Layered  Canvas.pdf"  
    if Path(local_path).exists():
        try:
            result = read_pdf_with_metadata.invoke({"source": local_path, "clean": True})
            print(f"\n✓ Local file read successfully!")
            print(f"  Pages: {result['num_pages']}")
            print(f"  Characters: {result['char_count']}")
        except Exception as e:
            print(f"✗ Failed: {e}")
    else:
        print(f"Local file not found: {local_path}")