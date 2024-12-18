import os
import json
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from Bio import Entrez
from Bio import Medline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_pubmed_articles(
    start_date: str,
    end_date: str,
    max_articles: int = 100
    ) -> List[Dict]:
    """
    Fetches PubMed articles within a date range.

    Args:
        start_date (str): Start date in format 'YYYY/MM/DD'.
        end_date (str): End date in format 'YYYY/MM/DD'.
        max_articles (int): Maximum number of articles to fetch.
        email (str): User email address (required by NCBI).
        api_key (str, optional): NCBI API key to increase rate limits.

    Returns:
        List[Dict]: A list of articles with their metadata.
    """
    # Retrieve Entrez email and API key from environment variables
    email = os.environ.get("ENTREZ_EMAIL")
    api_key = os.environ.get("ENTREZ_API_KEY")  # optional to increase rate limits

    if not email:
        raise ValueError("ENTREZ_EMAIL environment variable is not set.")
    
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # Build search term for date range
    search_term = f'"{start_date}"[Date - Publication] : "{end_date}"[Date - Publication]'

    logger.info(f"Searching PubMed for articles from {start_date} to {end_date}")
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_articles)
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]
    total_count = int(record["Count"])
    logger.info(f"Found {total_count} articles in the date range")

    # Adjust max_articles if fewer articles are available
    max_articles = min(max_articles, total_count)
    logger.info(f"Fetching up to {max_articles} articles")

    articles = []
    batch_size = 100  # NCBI recommends fetching records in batches
    for start in range(0, max_articles, batch_size):
        end = min(max_articles, start + batch_size)
        batch_ids = id_list[start:end]
        logger.info(f"Fetching records {start + 1} to {end}")
        fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
        records = Medline.parse(fetch_handle)
        for record in records:
            article = parse_medline_record(record)
            if article:
                articles.append(article)
        fetch_handle.close()
        time.sleep(0.5)  # To respect NCBI rate limits!

    logger.info(f"Fetched {len(articles)} articles with abstracts")
    return articles

def parse_medline_record(record) -> Optional[Dict]:
    """Parses a Medline record into the desired format."""
    pmid = record.get('PMID')
    title = record.get('TI') or record.get('Title')
    abstract = record.get('AB') or record.get('Abstract')
    authors = record.get('AU') or []
    journal = record.get('JT') or record.get('TA') or record.get('Journal')
    year = None
    if 'DP' in record:
        year_match = record['DP'][:4]
        if year_match.isdigit():
            year = int(year_match)

    if not abstract:
        logger.warning(f"No abstract found for PMID {pmid}, skipping")
        return None

    article = {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "year": year
    }
    return article  # parsed article or None if abstract missing

def fetch_and_save_articles(
        start_date: str,
        end_date: str,
        max_articles: int,
        output_file: str
    ) -> None:
    """Fetches articles and saves them to a JSON file."""
    articles = fetch_pubmed_articles(start_date, end_date, max_articles=max_articles)
    if articles:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Save articles to data.json
        with open(output_file, 'w') as f:
            json.dump(articles, f, indent=4)
        logger.info(f"Saved {len(articles)} articles to {output_file}")
    else:
        logger.info("No articles were fetched.")