from xml.etree import ElementTree as ET
from rpextractor.utils.logger import get_logger
from rpextractor.utils.config import load_yaml

logger = get_logger(__name__)

class XMLParser:
    """Prases XML files and extracts text countent"""

    def __init__(self):
        config = load_yaml("parser.yaml")
        self.body_sections = config.get("body_sections", {})
        self.special_sections = config.get("special_sections", {})
        logger.info(f"XMLParser initialized with body_sections: {self.body_sections} and special_sections: {self.special_sections}")

    def _extract_text(self, element) -> str:
        """Recursively extract text from an XML element."""
        if element is None: 
            return ""
        
        text_content = []

        if element.text:
            text_content.append(element.text.strip())
        
        for child in element:
            child_text = self._extract_text(child)
        
            if child_text:
                text_content.append(child_text)
            
            if child.tail:
                text_content.append(child.tail.strip())
                
        return " ".join(text_content).strip()

    def extract_metadata(self, root) -> dict:
        """Extract paper metadata from the front section."""
        metadata = {}
 
        title_elem = root.find(".//article-title")
        metadata["title"] = self._extract_text(title_elem) if title_elem is not None else ""
 
        pmcid_elem = root.find('.//article-id[@pub-id-type="pmcid"]')
        metadata["pmcid"] = pmcid_elem.text if pmcid_elem is not None else ""
 
        pmid_elem = root.find('.//article-id[@pub-id-type="pmid"]')
        metadata["pmid"] = pmid_elem.text if pmid_elem is not None else ""
 
        doi_elem = root.find('.//article-id[@pub-id-type="doi"]')
        metadata["doi"] = doi_elem.text if doi_elem is not None else ""
 
        authors = []
        for contrib in root.findall('.//contrib[@contrib-type="author"]'):
            surname = contrib.find(".//surname")
            given = contrib.find(".//given-names")
            if surname is not None and surname.text:
                name = surname.text
                if given is not None and given.text:
                    name = f"{given.text} {surname.text}"
                authors.append(name)
        metadata["authors"] = authors
 
        year_elem = root.find(".//pub-date/year")
        metadata["year"] = year_elem.text if year_elem is not None else ""
 
        keywords = []
        for kwd in root.findall(".//kwd-group//kwd"):
            if kwd.text:
                keywords.append(kwd.text)
        metadata["keywords"] = keywords
 
        logger.info(f"Extracted metadata for {metadata['pmcid']}")
        return metadata
    
    def extract_abstract(self, root) -> str:
        """Extract abstract text."""
        abstract = root.find(".//abstract")
        if abstract is None:
            logger.warning("Abstract not found")
            return ""
 
        paragraphs = []
 
        for sec in abstract.findall(".//sec"):
            title = sec.find("title")
            if title is not None and title.text:
                paragraphs.append(title.text)
            for para in sec.findall(".//p"):
                text = self._extract_text(para)
                if text:
                    paragraphs.append(text)
 
        if not paragraphs:
            for para in abstract.findall(".//p"):
                text = self._extract_text(para)
                if text:
                    paragraphs.append(text)
 
        return "\n".join(paragraphs)
 
    def extract_body_section(self, root, section_titles: list) -> str:
        """Extract a section from the body by matching title variants."""
        body = root.find(".//body")
        if body is None:
            return ""
 
        section_titles = [t.lower() for t in section_titles]
 
        for sec in body.findall(".//sec"):
            title_elem = sec.find("title")
            if title_elem is None or not title_elem.text:
                continue
 
            title_text = title_elem.text.strip().lower()
 
            if any(s in title_text for s in section_titles):
                parts = [title_elem.text]
 
                for para in sec.findall(".//p"):
                    text = self._extract_text(para)
                    if text:
                        parts.append(text)
 
                for subsec in sec.findall(".//sec"):
                    subtitle = subsec.find("title")
                    if subtitle is not None and subtitle.text:
                        parts.append(subtitle.text)
                    for para in subsec.findall(".//p"):
                        text = self._extract_text(para)
                        if text:
                            parts.append(text)
 
                return "\n".join(parts)
 
        return ""
 
    def extract_data_availability(self, root) -> str:
        """Extract data availability statement from body, back, or notes."""
        titles = self.special_sections.get("DATA_AVAILABILITY", [])
        titles = [t.lower() for t in titles]
 
        for elem in root.findall(".//*[title]"):
            title = elem.find("title")
            if title is not None and title.text:
                if any(t in title.text.strip().lower() for t in titles):
                    paragraphs = []
                    for para in elem.findall(".//p"):
                        text = self._extract_text(para)
                        if text:
                            paragraphs.append(text)
                    if paragraphs:
                        return "\n".join(paragraphs)
 
        return ""
 
    def extract_supplementary(self, root) -> str:
        """Extract supplementary material information."""
        titles = self.special_sections.get("ASSOCIATED_DATA", [])
        parts = []
 
        for supp in root.findall(".//supplementary-material"):
            label = supp.find("label")
            caption = supp.find("caption")
 
            if label is not None and label.text:
                parts.append(label.text)
            if caption is not None:
                text = self._extract_text(caption)
                if text:
                    parts.append(text)
 
            media = supp.find(".//media")
            if media is not None:
                href = media.get("{http://www.w3.org/1999/xlink}href")
                if href:
                    parts.append(f"File: {href}")
 
        titles_lower = [t.lower() for t in titles]
        for sec in root.findall(".//sec"):
            title = sec.find("title")
            if title is not None and title.text:
                if any(t in title.text.lower() for t in titles_lower):
                    parts.append(title.text)
                    for para in sec.findall(".//p"):
                        text = self._extract_text(para)
                        if text:
                            parts.append(text)
 
        return "\n".join(parts) if parts else ""
 
    def extract_tables(self, root) -> str:
        """Extract table content. Reserved for future use."""
        parts = []
 
        for table_wrap in root.findall(".//table-wrap"):
            label = table_wrap.find("label")
            caption = table_wrap.find("caption")
 
            if label is not None and label.text:
                parts.append(label.text)
            if caption is not None:
                text = self._extract_text(caption)
                if text:
                    parts.append(text)
 
            for row in table_wrap.findall(".//tr"):
                cells = []
                for cell in row.findall(".//*"):
                    if cell.tag in ("th", "td"):
                        text = self._extract_text(cell)
                        if text:
                            cells.append(text)
                if cells:
                    parts.append(" | ".join(cells))
 
        return "\n".join(parts) if parts else ""
 
    def extract_figures(self, root) -> str:
        """Extract figure captions. Reserved for future use."""
        parts = []
 
        for fig in root.findall(".//fig"):
            label = fig.find("label")
            caption = fig.find("caption")
 
            if label is not None and label.text:
                parts.append(label.text)
            if caption is not None:
                text = self._extract_text(caption)
                if text:
                    parts.append(text)
 
        return "\n".join(parts) if parts else ""
 
    def parse(self, xml_path: str) -> dict:
        """Parse a single XML file and return structured content.
 
        Args:
            xml_path: Path to the XML file in data/raw/
 
        Returns:
            Dict with metadata and all extracted sections.
            Returns None if parsing fails.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
 
            article = root.find(".//article")
            if article is None:
                article = root
 
            metadata = self.extract_metadata(article)
            logger.info(f"Parsing {metadata.get('pmcid', xml_path)}")
 
            content = {"metadata": metadata}
 
            content["abstract"] = self.extract_abstract(article)
 
            for section_name, title_variants in self.body_sections.items():
                content[section_name.lower()] = self.extract_body_section(article, title_variants)
 
            content["data_availability"] = self.extract_data_availability(article)
            content["supplementary"] = self.extract_supplementary(article)
            content["tables"] = self.extract_tables(article)
            content["figures"] = self.extract_figures(article)
 
            logger.info(f"Successfully parsed {metadata.get('pmcid', xml_path)}")
            return content
 
        except ET.ParseError as e:
            logger.error(f"XML parse error for {xml_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {xml_path}: {e}")
            return None