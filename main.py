import json
import os
import logging
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm

from src.models.cerebras_inference import CerebrasInference
from src.knowledge_graph_updater import KnowledgeGraphUpdater
from src.pubmed_scraper import fetch_and_save_articles

logger = logging.getLogger(__name__)

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'knowledge_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )
    return log_file

log_file = setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # parse CL args
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", default="./data/knowledge_graph.json",
                        help="Path to the knowledge graph JSON file.")
    parser.add_argument("--data_path", default="./data/data.json",
                        help="Path to the input abstracts JSON file.")
    # Can switch back to llama3.1-8b for lighter compute
    parser.add_argument("--model_name", default="llama3.3-70b",
                        help="Name of the model to use")
    # Additional args for article fetching:
    parser.add_argument("--max_articles", type=int, default=100, help="Maximum number of articles to fetch")
    parser.add_argument("--years_back", type=int, default=1, help="How many years back to fetch articles")
    args = parser.parse_args()
    
    # Logging
    print("Program started.")
    print(f"Model name: {args.model_name}")
    logger.info(f"Logging to file: {log_file}")

    # If data.json doesn't exist / is empty, fetch new articles
    if not os.path.exists(args.data_path) or os.path.getsize(args.data_path) == 0:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.years_back*365)
        start_date_str = start_date.strftime('%Y/%m/%d')
        end_date_str = end_date.strftime('%Y/%m/%d')

        logger.info(f"No or empty data file found at {args.data_path}. Fetching new articles...")
        fetch_and_save_articles(
            start_date=start_date_str,
            end_date=end_date_str,
            max_articles=args.max_articles,
            output_file=args.data_path
        )

    os.makedirs("data", exist_ok=True)

    # Initialize CerebrasInference
    inference = CerebrasInference(model=args.model_name)

    # Initialize KnowledgeGraphUpdater
    updater = KnowledgeGraphUpdater(
        graph_path=args.graph_path,
        entity_aliases_path="./data/entity_aliases.json",
        model=inference
    )

    # Load abstracts
    with open(args.data_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} abstracts for processing.")

    # Use tqdm for progress tracking
    for abstract_info in tqdm(data, desc="Processing Abstracts", unit="abstract"):
        try:
            print(f"\nProcessing abstract with PMID: {abstract_info.get('pmid', 'N/A')}")
            print(f"Title: {abstract_info['title']}")
            print(f"Abstract: {abstract_info['abstract'][:100]}...")

            # process abstract + update graph
            updates = updater.process_abstract(abstract_info)
            logger.info(f"Successfully processed abstract {abstract_info.get('pmid', 'N/A')} with {len(updates)} updates.")
            updater.save_graph()  # save updated graph
            logger.info("Successfully saved updated knowledge graph")
        except Exception as e:
            logger.error(f"Error processing abstract {abstract_info.get('pmid', 'N/A')}: {e}")

    logger.info("Finished processing all abstracts.")
    print("Program completed.")