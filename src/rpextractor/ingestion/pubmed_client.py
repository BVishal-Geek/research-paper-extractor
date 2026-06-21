import os
from Bio import Entrez
from dotenv import load_dotenv
from rpextractor.utils.logger import get_logger
from rpextractor.utils.config import load_yaml

load_dotenv()
# file_name = os.path.basename(__file__)
logger = get_logger(__name__)

class PubMedClient:
    """Client for interacting with PubMed Entrez API.
    
    Handles searching for papers and fetching XML content.
    Credentials loaded from .env, config from pubmed.yaml.
    """
    def __init__(self):
        config = load_yaml("pubmed.yaml")
        self.email = os.getenv("EMAIL")
        self.api_key = os.getenv("API_KEY_PUBMED")
        if not self.email or not self.api_key:
            logger.error("PUBMED_EMAIL and PUBMED_API_KEY must be set in .env")
            raise ValueError("PUBMED_EMAIL and PUBMED_API_KEY must be set in .env")
        Entrez.email = self.email
        Entrez.api_key = self.api_key
        self.query = config.get("queries")
        self.max_results = config.get("max_results", 100)
        self.database = config.get("database", "pubmed")
        self.retmode = config.get("retmode", "xml")
        self.datetype = config.get("datetype", "pdat")
        self.mindate = config.get("mindate", "2020/01/01")
        self.sort = config.get("sort", "relevance")
        self.rettype = config.get("rettype", "full")

    def search(self, query: str = None) -> list[str]:
        """Search PubMed with the given query and return a list of PMIDs."""
        if query is None:
            query = self.query

        query  = f"{query} AND {self.mindate}[{self.datetype}]"
        logger.info(f"Using default query from config : {query}")
        try:
            handle = Entrez.esearch(
                db=self.database,
                term=query,
                retmax=self.max_results,
                sort=self.sort,
                retmode=self.retmode,
                usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()
            pmids = record.get("IdList")
            total_count = record.get("Count")
            logger.info(f"Search completed. Found {total_count} results, returning {len(pmids)} PMIDs.")
            return pmids
        
        except Exception as e:
            logger.error(f"Error during PubMed search: {e}")
            return []
    
    def fetch(self, pmid: str) -> str:
        """ Fetch the XML data for the given list"""
        if not pmid:
            logger.warning("No PMIDs provided to fetch.")
            return ""
        pmc_id = pmid.replace("PMC", "")
        try:
            handle = Entrez.efetch(
                db=self.database,
                id=pmc_id,
                retmode=self.retmode,
                rettype=self.rettype
            )
            xml_data = handle.read().decode("utf-8")
            handle.close()
            logger.info(f"Fetched XML data for {len(pmc_id)} PMIDs.")
            return xml_data
        except Exception as e:
            logger.error(f"Error during PubMed fetch: {e}")
            return ""
        

    
