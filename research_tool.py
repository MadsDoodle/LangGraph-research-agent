"""
Multi-Research Database Search Tools
Supports: arXiv, PubMed, Semantic Scholar, CrossRef
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from langchain_core.tools import tool
import logging
from datetime import datetime

import os
from pathlib import Path
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'research_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ==================== arXiv Search ====================
def search_arxiv_papers(topic: str, max_results: int = 5) -> dict:
    """Search arXiv papers using their API"""
    logger.info(f"[arXiv] Starting search for topic: '{topic}' with max_results: {max_results}")
    
    query = "+".join(topic.lower().split())
    logger.debug(f"[arXiv] Formatted query: '{query}'")
    
    for char in list('()" '):
        if char in query:
            logger.error(f"[arXiv] Invalid character '{char}' found in query: {query}")
            raise ValueError(f"Cannot have character: '{char}' in query: {query}")

    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{query}"
        f"&max_results={max_results}"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
    )
    logger.info(f"[arXiv] API URL: {url}")
    
    try:
        resp = requests.get(url, timeout=30)
        logger.debug(f"[arXiv] Response status code: {resp.status_code}")
        
        if not resp.ok:
            logger.error(f"[arXiv] API request failed: {resp.status_code} - {resp.text[:200]}")
            raise ValueError(f"Bad response from arXiv API: {resp}\n{resp.text}")
        
        logger.info(f"[arXiv] Successfully received response, parsing XML...")
        data = parse_arxiv_xml(resp.text)
        logger.info(f"[arXiv] Parsing complete. Found {len(data['entries'])} papers")
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[arXiv] Request exception: {e}")
        raise
    except Exception as e:
        logger.error(f"[arXiv] Unexpected error: {e}")
        raise


def parse_arxiv_xml(xml_content: str) -> dict:
    """Parse the XML content from arXiv API response"""
    logger.info("[arXiv] Starting XML parsing")
    logger.debug(f"[arXiv] XML content length: {len(xml_content)} characters")
    
    entries = []
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    try:
        root = ET.fromstring(xml_content)
        logger.debug(f"[arXiv] XML root element: {root.tag}")
        
        entry_elements = root.findall("atom:entry", ns)
        logger.info(f"[arXiv] Found {len(entry_elements)} entry elements")
        
        for idx, entry in enumerate(entry_elements, 1):
            logger.debug(f"[arXiv] Processing entry {idx}/{len(entry_elements)}")
            
            authors = [
                author.findtext("atom:name", namespaces=ns)
                for author in entry.findall("atom:author", ns)
            ]
            logger.debug(f"[arXiv] Entry {idx}: Found {len(authors)} authors")
            
            categories = [
                cat.attrib.get("term")
                for cat in entry.findall("atom:category", ns)
            ]
            logger.debug(f"[arXiv] Entry {idx}: Found {len(categories)} categories")
            
            pdf_link = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("type") == "application/pdf":
                    pdf_link = link.attrib.get("href")
                    break
            
            title = entry.findtext("atom:title", namespaces=ns)
            logger.debug(f"[arXiv] Entry {idx}: Title = '{title[:50]}...'")

            entries.append({
                "title": title,
                "summary": entry.findtext("atom:summary", namespaces=ns).strip(),
                "authors": authors,
                "categories": categories,
                "pdf": pdf_link,
                "source": "arXiv"
            })
        
        logger.info(f"[arXiv] Successfully parsed {len(entries)} entries")
        return {"entries": entries}
        
    except ET.ParseError as e:
        logger.error(f"[arXiv] XML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"[arXiv] Unexpected parsing error: {e}")
        raise


# ==================== PubMed Search ====================
def search_pubmed_papers(topic: str, max_results: int = 5) -> dict:
    """Search PubMed papers using NCBI E-utilities API"""
    logger.info(f"[PubMed] Starting search for topic: '{topic}' with max_results: {max_results}")
    
    # Step 1: Search for paper IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": topic,
        "retmax": max_results,
        "retmode": "json",
        "sort": "pub_date"
    }
    
    logger.info(f"[PubMed] Step 1: Searching for paper IDs")
    logger.debug(f"[PubMed] Search URL: {search_url}")
    logger.debug(f"[PubMed] Search params: {search_params}")
    
    try:
        search_resp = requests.get(search_url, params=search_params, timeout=30)
        logger.debug(f"[PubMed] Search response status: {search_resp.status_code}")
        
        if not search_resp.ok:
            logger.error(f"[PubMed] Search request failed: {search_resp.status_code}")
            raise ValueError(f"PubMed search failed: {search_resp.status_code}")
        
        search_data = search_resp.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"[PubMed] Found {len(id_list)} paper IDs: {id_list}")
        
        if not id_list:
            logger.warning(f"[PubMed] No papers found for topic: '{topic}'")
            return {"entries": []}
        
        # Step 2: Fetch details for each paper
        logger.info(f"[PubMed] Step 2: Fetching paper details")
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        logger.debug(f"[PubMed] Fetch URL: {fetch_url}")
        logger.debug(f"[PubMed] Fetch params: {fetch_params}")
        
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=30)
        logger.debug(f"[PubMed] Fetch response status: {fetch_resp.status_code}")
        
        if not fetch_resp.ok:
            logger.error(f"[PubMed] Fetch request failed: {fetch_resp.status_code}")
            raise ValueError(f"PubMed fetch failed: {fetch_resp.status_code}")
        
        logger.info(f"[PubMed] Successfully fetched paper details, parsing...")
        data = parse_pubmed_json(fetch_resp.json())
        logger.info(f"[PubMed] Parsing complete. Returning {len(data['entries'])} papers")
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[PubMed] Request exception: {e}")
        raise
    except Exception as e:
        logger.error(f"[PubMed] Unexpected error: {e}")
        raise


def parse_pubmed_json(json_data: dict) -> dict:
    """Parse PubMed JSON response"""
    logger.info("[PubMed] Starting JSON parsing")
    entries = []
    results = json_data.get("result", {})
    logger.debug(f"[PubMed] Result keys: {list(results.keys())}")
    
    paper_count = 0
    for pmid, paper in results.items():
        if pmid == "uids":
            continue
        
        paper_count += 1
        logger.debug(f"[PubMed] Processing paper {paper_count}: PMID={pmid}")
        
        authors = []
        if "authors" in paper:
            authors = [author.get("name", "") for author in paper["authors"]]
            logger.debug(f"[PubMed] Paper {pmid}: Found {len(authors)} authors")
        
        title = paper.get("title", "")
        logger.debug(f"[PubMed] Paper {pmid}: Title = '{title[:50]}...'")
        
        entries.append({
            "title": title,
            "summary": paper.get("abstract", "No abstract available"),
            "authors": authors,
            "categories": [paper.get("source", "")],
            "pdf": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "pubmed_id": pmid,
            "publication_date": paper.get("pubdate", ""),
            "source": "PubMed"
        })
    
    logger.info(f"[PubMed] Successfully parsed {len(entries)} entries")
    return {"entries": entries}


# ==================== Semantic Scholar Search ====================
def search_semantic_scholar_papers(topic: str, max_results: int = 5) -> dict:
    """Search Semantic Scholar papers using their API"""
    logger.info(f"[Semantic Scholar] Starting search for topic: '{topic}' with max_results: {max_results}")
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": topic,
        "limit": max_results,
        "fields": "title,abstract,authors,year,citationCount,url,openAccessPdf"
    }
    
    logger.info(f"[Semantic Scholar] API URL: {url}")
    logger.debug(f"[Semantic Scholar] Params: {params}")
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        logger.debug(f"[Semantic Scholar] Response status: {resp.status_code}")
        
        if not resp.ok:
            logger.error(f"[Semantic Scholar] Request failed: {resp.status_code} - {resp.text[:200]}")
            raise ValueError(f"Semantic Scholar search failed: {resp.status_code}")
        
        logger.info(f"[Semantic Scholar] Successfully received response, parsing...")
        data = parse_semantic_scholar_json(resp.json())
        logger.info(f"[Semantic Scholar] Parsing complete. Found {len(data['entries'])} papers")
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[Semantic Scholar] Request exception: {e}")
        raise
    except Exception as e:
        logger.error(f"[Semantic Scholar] Unexpected error: {e}")
        raise


def parse_semantic_scholar_json(json_data: dict) -> dict:
    """Parse Semantic Scholar JSON response"""
    logger.info("[Semantic Scholar] Starting JSON parsing")
    entries = []
    
    papers = json_data.get("data", [])
    logger.info(f"[Semantic Scholar] Found {len(papers)} papers in response")
    
    for idx, paper in enumerate(papers, 1):
        logger.debug(f"[Semantic Scholar] Processing paper {idx}/{len(papers)}")
        
        authors = [author.get("name", "") for author in paper.get("authors", [])]
        logger.debug(f"[Semantic Scholar] Paper {idx}: Found {len(authors)} authors")
        
        pdf_link = None
        if paper.get("openAccessPdf"):
            pdf_link = paper["openAccessPdf"].get("url")
            logger.debug(f"[Semantic Scholar] Paper {idx}: PDF available = {pdf_link is not None}")
        
        title = paper.get("title", "")
        citation_count = paper.get("citationCount", 0)
        logger.debug(f"[Semantic Scholar] Paper {idx}: '{title[:50]}...' (Citations: {citation_count})")
        
        entries.append({
            "title": title,
            "summary": paper.get("abstract", "No abstract available"),
            "authors": authors,
            "categories": [],
            "pdf": pdf_link,
            "url": paper.get("url", ""),
            "year": paper.get("year", ""),
            "citation_count": citation_count,
            "source": "Semantic Scholar"
        })
    
    logger.info(f"[Semantic Scholar] Successfully parsed {len(entries)} entries")
    return {"entries": entries}


# ==================== CrossRef Search ====================
def search_crossref_papers(topic: str, max_results: int = 5) -> dict:
    """Search CrossRef papers using their API"""
    logger.info(f"[CrossRef] Starting search for topic: '{topic}' with max_results: {max_results}")
    
    url = "https://api.crossref.org/works"
    params = {
        "query": topic,
        "rows": max_results,
        "sort": "published",
        "order": "desc"
    }
    
    logger.info(f"[CrossRef] API URL: {url}")
    logger.debug(f"[CrossRef] Params: {params}")
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        logger.debug(f"[CrossRef] Response status: {resp.status_code}")
        
        if not resp.ok:
            logger.error(f"[CrossRef] Request failed: {resp.status_code} - {resp.text[:200]}")
            raise ValueError(f"CrossRef search failed: {resp.status_code}")
        
        logger.info(f"[CrossRef] Successfully received response, parsing...")
        data = parse_crossref_json(resp.json())
        logger.info(f"[CrossRef] Parsing complete. Found {len(data['entries'])} papers")
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[CrossRef] Request exception: {e}")
        raise
    except Exception as e:
        logger.error(f"[CrossRef] Unexpected error: {e}")
        raise


def parse_crossref_json(json_data: dict) -> dict:
    """Parse CrossRef JSON response"""
    logger.info("[CrossRef] Starting JSON parsing")
    entries = []
    
    items = json_data.get("message", {}).get("items", [])
    logger.info(f"[CrossRef] Found {len(items)} items in response")
    
    for idx, item in enumerate(items, 1):
        logger.debug(f"[CrossRef] Processing item {idx}/{len(items)}")
        
        authors = []
        for author in item.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)
        logger.debug(f"[CrossRef] Item {idx}: Found {len(authors)} authors")
        
        title = " ".join(item.get("title", []))
        abstract = " ".join(item.get("abstract", ["No abstract available"]))
        
        # Get DOI link
        doi = item.get("DOI", "")
        url = f"https://doi.org/{doi}" if doi else ""
        publisher = item.get("publisher", "")
        
        logger.debug(f"[CrossRef] Item {idx}: Title = '{title[:50]}...'")
        logger.debug(f"[CrossRef] Item {idx}: DOI = {doi}, Publisher = {publisher}")
        
        entries.append({
            "title": title,
            "summary": abstract,
            "authors": authors,
            "categories": item.get("subject", []),
            "pdf": url,
            "doi": doi,
            "publisher": publisher,
            "source": "CrossRef"
        })
    
    logger.info(f"[CrossRef] Successfully parsed {len(entries)} entries")
    return {"entries": entries}

def download_paper_pdf(pdf_url: str, title: str, source: str, output_base_dir: str = "downloaded_papers") -> Optional[str]:
    """Download a paper PDF and save it to source-specific directory
    
    Args:
        pdf_url: URL of the PDF to download
        title: Title of the paper (used for filename)
        source: Source database (arXiv, PubMed, etc.)
        output_base_dir: Base directory for all downloads
        
    Returns:
        Path to downloaded file or None if download failed
    """
    if not pdf_url:
        logger.warning(f"[Download] No PDF URL provided for '{title[:50]}'")
        return None
    
    logger.info(f"[Download] Starting download from {source}: '{title[:50]}...'")
    
    # Create source-specific directory
    source_dir = Path(output_base_dir) / source.lower().replace(" ", "_")
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean title for filename (remove invalid characters)
    clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
    clean_title = clean_title[:100]  # Limit filename length
    filename = f"{clean_title}.pdf"
    filepath = source_dir / filename
    
    logger.debug(f"[Download] Target filepath: {filepath}")
    
    try:
        response = requests.get(pdf_url, timeout=60, stream=True)
        
        if not response.ok:
            logger.error(f"[Download] Failed to download: {response.status_code}")
            return None
        
        # Write PDF to file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"[Download] Successfully saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"[Download] Error downloading PDF: {e}")
        return None


# ==================== LangChain Tools ====================
@tool
def arxiv_search(topic: str, max_results: int = 5) -> List[Dict]:
    """Search for recently uploaded arXiv papers

    Args:
        topic: The topic to search for papers about
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of papers with their metadata including title, authors, summary, etc.
    """
    logger.info("=" * 80)
    logger.info("[TOOL] arxiv_search called")
    logger.info(f"[TOOL] Parameters: topic='{topic}', max_results={max_results}")
    
    try:
        papers = search_arxiv_papers(topic, max_results)
        
        if len(papers["entries"]) == 0:
            logger.warning(f"[TOOL] No papers found for topic: {topic}")
            raise ValueError(f"No papers found for topic: {topic}")
        
        logger.info(f"[TOOL] Successfully retrieved {len(papers['entries'])} papers")
        logger.info("=" * 80)
        return papers["entries"]
        
    except Exception as e:
        logger.error(f"[TOOL] arxiv_search failed: {e}")
        logger.info("=" * 80)
        raise


@tool
def pubmed_search(topic: str, max_results: int = 5) -> List[Dict]:
    """Search for biomedical and life sciences papers on PubMed

    Args:
        topic: The topic to search for papers about
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of papers with their metadata including title, authors, summary, etc.
    """
    logger.info("=" * 80)
    logger.info("[TOOL] pubmed_search called")
    logger.info(f"[TOOL] Parameters: topic='{topic}', max_results={max_results}")
    
    try:
        papers = search_pubmed_papers(topic, max_results)
        
        if len(papers["entries"]) == 0:
            logger.warning(f"[TOOL] No papers found for topic: {topic}")
            raise ValueError(f"No papers found for topic: {topic}")
        
        logger.info(f"[TOOL] Successfully retrieved {len(papers['entries'])} papers")
        logger.info("=" * 80)
        return papers["entries"]
        
    except Exception as e:
        logger.error(f"[TOOL] pubmed_search failed: {e}")
        logger.info("=" * 80)
        raise


@tool
def semantic_scholar_search(topic: str, max_results: int = 5) -> List[Dict]:
    """Search for papers on Semantic Scholar (AI/CS focused)

    Args:
        topic: The topic to search for papers about
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of papers with their metadata including title, authors, summary, citation count, etc.
    """
    logger.info("=" * 80)
    logger.info("[TOOL] semantic_scholar_search called")
    logger.info(f"[TOOL] Parameters: topic='{topic}', max_results={max_results}")
    
    try:
        papers = search_semantic_scholar_papers(topic, max_results)
        
        if len(papers["entries"]) == 0:
            logger.warning(f"[TOOL] No papers found for topic: {topic}")
            raise ValueError(f"No papers found for topic: {topic}")
        
        logger.info(f"[TOOL] Successfully retrieved {len(papers['entries'])} papers")
        logger.info("=" * 80)
        return papers["entries"]
        
    except Exception as e:
        logger.error(f"[TOOL] semantic_scholar_search failed: {e}")
        logger.info("=" * 80)
        raise


@tool
def crossref_search(topic: str, max_results: int = 5) -> List[Dict]:
    """Search for papers across multiple publishers using CrossRef

    Args:
        topic: The topic to search for papers about
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of papers with their metadata including title, authors, summary, DOI, etc.
    """
    logger.info("=" * 80)
    logger.info("[TOOL] crossref_search called")
    logger.info(f"[TOOL] Parameters: topic='{topic}', max_results={max_results}")
    
    try:
        papers = search_crossref_papers(topic, max_results)
        
        if len(papers["entries"]) == 0:
            logger.warning(f"[TOOL] No papers found for topic: {topic}")
            raise ValueError(f"No papers found for topic: {topic}")
        
        logger.info(f"[TOOL] Successfully retrieved {len(papers['entries'])} papers")
        logger.info("=" * 80)
        return papers["entries"]
        
    except Exception as e:
        logger.error(f"[TOOL] crossref_search failed: {e}")
        logger.info("=" * 80)
        raise


@tool
def multi_database_search(topic: str, max_results: int = 5) -> Dict[str, List[Dict]]:
    """Search multiple research databases simultaneously

    Args:
        topic: The topic to search for papers about
        max_results: Maximum number of results per database (default: 5)

    Returns:
        Dictionary with results from each database
    """
    logger.info("=" * 80)
    logger.info("[TOOL] multi_database_search called")
    logger.info(f"[TOOL] Parameters: topic='{topic}', max_results={max_results}")
    logger.info("[TOOL] Will search across: arXiv, PubMed, Semantic Scholar, CrossRef")
    
    results = {}
    
    databases = {
        "arxiv": search_arxiv_papers,
        "pubmed": search_pubmed_papers,
        "semantic_scholar": search_semantic_scholar_papers,
        "crossref": search_crossref_papers
    }
    
    for db_name, search_func in databases.items():
        try:
            logger.info(f"[TOOL] Searching {db_name}...")
            data = search_func(topic, max_results)
            results[db_name] = data["entries"]
            logger.info(f"[TOOL] {db_name}: Retrieved {len(data['entries'])} papers")
        except Exception as e:
            logger.error(f"[TOOL] Error searching {db_name}: {e}")
            results[db_name] = []
    
    total_papers = sum(len(papers) for papers in results.values())
    logger.info(f"[TOOL] Multi-database search complete. Total papers: {total_papers}")
    logger.info("=" * 80)
    
    return results


@tool
def download_papers_from_search(topic: str, source: str = "arxiv", max_results: int = 3, output_dir: str = "downloaded_papers") -> Dict:
    """Search for papers and download their PDFs
    
    Args:
        topic: The topic to search for papers about
        source: Database to search ('arxiv', 'pubmed', 'semantic_scholar', 'crossref', or 'all')
        max_results: Maximum number of papers to download per source
        output_dir: Base directory to save downloaded PDFs
        
    Returns:
        Dictionary with download results and statistics
    """
    logger.info("=" * 80)
    logger.info("[TOOL] download_papers_from_search called")
    logger.info(f"[TOOL] Parameters: topic='{topic}', source='{source}', max_results={max_results}")
    
    search_functions = {
        "arxiv": search_arxiv_papers,
        "pubmed": search_pubmed_papers,
        "semantic_scholar": search_semantic_scholar_papers,
        "crossref": search_crossref_papers
    }
    
    sources_to_search = search_functions.keys() if source == "all" else [source]
    
    results = {
        "successful_downloads": [],
        "failed_downloads": [],
        "total_papers_found": 0,
        "total_downloaded": 0
    }
    
    for src in sources_to_search:
        if src not in search_functions:
            logger.warning(f"[TOOL] Unknown source: {src}")
            continue
            
        try:
            logger.info(f"[TOOL] Searching {src}...")
            papers_data = search_functions[src](topic, max_results)
            papers = papers_data["entries"]
            results["total_papers_found"] += len(papers)
            
            for paper in papers:
                pdf_url = paper.get("pdf") or paper.get("url")
                if pdf_url:
                    filepath = download_paper_pdf(
                        pdf_url=pdf_url,
                        title=paper["title"],
                        source=paper["source"],
                        output_base_dir=output_dir
                    )
                    
                    if filepath:
                        results["successful_downloads"].append({
                            "title": paper["title"],
                            "source": paper["source"],
                            "filepath": filepath
                        })
                        results["total_downloaded"] += 1
                    else:
                        results["failed_downloads"].append({
                            "title": paper["title"],
                            "source": paper["source"],
                            "reason": "Download failed"
                        })
                else:
                    results["failed_downloads"].append({
                        "title": paper["title"],
                        "source": paper["source"],
                        "reason": "No PDF URL available"
                    })
                    
        except Exception as e:
            logger.error(f"[TOOL] Error processing {src}: {e}")
    
    logger.info(f"[TOOL] Download complete: {results['total_downloaded']}/{results['total_papers_found']} papers")
    logger.info("=" * 80)
    
    return results


# ==================== Example Usage ====================
if __name__ == "__main__":
    logger.info("*" * 80)
    logger.info("STARTING RESEARCH SEARCH TOOL TESTS")
    logger.info("*" * 80)

    # Take user input for the topic
    topic = input("Enter the research topic you want to search for: ").strip()
    if not topic:
        print("No topic entered. Using default: 'machine learning transformers'")
        topic = "machine learning transformers"

    print("\n" + "=" * 80)
    print("=== Testing arXiv ===")
    print("=" * 80)
    try:
        logger.info(f"Testing arXiv search with topic: '{topic}'")
        arxiv_results = arxiv_search.invoke({"topic": topic, "max_results": 3})
        print(f"\n✓ Found {len(arxiv_results)} arXiv papers")
        for i, paper in enumerate(arxiv_results, 1):
            print(f"  {i}. {paper['title'][:80]}...")
            print(f"     Authors: {', '.join(paper['authors'][:3])}")
            print(f"     PDF: {paper['pdf']}\n")
    except Exception as e:
        logger.error(f"arXiv test failed: {e}")
        print(f"✗ arXiv test failed: {e}")

    print("\n" + "=" * 80)
    print("=== Testing PubMed ===")
    print("=" * 80)
    try:
        logger.info(f"Testing PubMed search with topic: '{topic}'")
        pubmed_results = pubmed_search.invoke({"topic": topic, "max_results": 3})
        print(f"\n✓ Found {len(pubmed_results)} PubMed papers")
        for i, paper in enumerate(pubmed_results, 1):
            print(f"  {i}. {paper['title'][:80]}...")
            print(f"     Authors: {', '.join(paper['authors'][:3])}")
            print(f"     PMID: {paper.get('pubmed_id', 'N/A')}\n")
    except Exception as e:
        logger.error(f"PubMed test failed: {e}")
        print(f"✗ PubMed test failed: {e}")

    print("\n" + "=" * 80)
    print("=== Testing Semantic Scholar ===")
    print("=" * 80)
    try:
        logger.info(f"Testing Semantic Scholar search with topic: '{topic}'")
        ss_results = semantic_scholar_search.invoke({"topic": topic, "max_results": 3})
        print(f"\n✓ Found {len(ss_results)} Semantic Scholar papers")
        for i, paper in enumerate(ss_results, 1):
            print(f"  {i}. {paper['title'][:80]}...")
            print(f"     Authors: {', '.join(paper['authors'][:3])}")
            print(f"     Citations: {paper.get('citation_count', 'N/A')}")
            print(f"     Year: {paper.get('year', 'N/A')}\n")
    except Exception as e:
        logger.error(f"Semantic Scholar test failed: {e}")
        print(f"✗ Semantic Scholar test failed: {e}")

    print("\n" + "=" * 80)
    print("=== Testing CrossRef ===")
    print("=" * 80)
    try:
        logger.info(f"Testing CrossRef search with topic: '{topic}'")
        crossref_results = crossref_search.invoke({"topic": topic, "max_results": 3})
        print(f"\n✓ Found {len(crossref_results)} CrossRef papers")
        for i, paper in enumerate(crossref_results, 1):
            print(f"  {i}. {paper['title'][:80]}...")
            print(f"     Authors: {', '.join(paper['authors'][:3])}")
            print(f"     Publisher: {paper.get('publisher', 'N/A')}")
            print(f"     DOI: {paper.get('doi', 'N/A')}\n")
    except Exception as e:
        logger.error(f"CrossRef test failed: {e}")
        print(f"✗ CrossRef test failed: {e}")

    print("\n" + "=" * 80)
    print("=== Testing Multi-Database Search ===")
    print("=" * 80)
    try:
        logger.info(f"Testing multi-database search with topic: '{topic}'")
        multi_results = multi_database_search.invoke({"topic": topic, "max_results": 2})
        print(f"\n✓ Multi-database search complete:")
        for db, papers in multi_results.items():
            print(f"  {db}: {len(papers)} papers")
            for i, paper in enumerate(papers, 1):
                print(f"    {i}. {paper['title'][:60]}...")

        total_papers = sum(len(papers) for papers in multi_results.values())
        print(f"\n  Total papers across all databases: {total_papers}")
    except Exception as e:
        logger.error(f"Multi-database test failed: {e}")
        print(f"✗ Multi-database test failed: {e}")

    logger.info("*" * 80)
    logger.info("ALL TESTS COMPLETED")
    logger.info("*" * 80)
    print("\n" + "=" * 80)
    print("Check the log file for detailed execution traces")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("=== Testing PDF Download ===")
    print("=" * 80)
    try:
        logger.info(f"Testing PDF download with topic: '{topic}'")
        download_results = download_papers_from_search.invoke({
            "topic": topic,
            "source": "arxiv",  # or "all" for all sources
            "max_results": 2,
            "output_dir": "research_papers"
        })
        print(f"\n✓ Downloaded {download_results['total_downloaded']}/{download_results['total_papers_found']} papers")
        print(f"  Successful: {len(download_results['successful_downloads'])}")
        print(f"  Failed: {len(download_results['failed_downloads'])}")
        
        if download_results['successful_downloads']:
            print("\n  Downloaded files:")
            for item in download_results['successful_downloads']:
                print(f"    - {item['filepath']}")
    except Exception as e:
        logger.error(f"PDF download test failed: {e}")
        print(f"✗ PDF download test failed: {e}")
