"""
Summary Extractor Module
========================
Extracts and cleans plain text content from HTML summaries, agenda descriptions,
and portal briefing sheets using standard library HTML parsers.
"""

import re
from html.parser import HTMLParser
from pathlib import Path


class SummaryHTMLParser(HTMLParser):
    """
    Standard HTML parser that strips scripts, style sections, and tags,
    retaining only clean, human-readable text.
    """

    def __init__(self):
        super().__init__()
        self.reset()
        self.convert_charrefs = True
        self.text_accumulator = []
        self.ignore_content = False
        self.ignored_tags = {"script", "style", "head", "meta", "link"}

    def handle_starttag(self, tag, attrs):
        if tag in self.ignored_tags:
            self.ignore_content = True
        # Inject spacer newlines for block elements to preserve word separation
        elif tag in {"p", "br", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self.text_accumulator.append("\n")

    def handle_endtag(self, tag):
        if tag in self.ignored_tags:
            self.ignore_content = False
        elif tag in {"p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self.text_accumulator.append("\n")

    def handle_data(self, data):
        if not self.ignore_content:
            self.text_accumulator.append(data)

    def get_clean_text(self) -> str:
        """
        Processes parsed content, removes excess spacing, and returns clean text.
        """
        raw_text = "".join(self.text_accumulator)
        # Collapse multiple whitespace and lines
        lines = [line.strip() for line in raw_text.splitlines()]
        cleaned_text = "\n".join([line for line in lines if line])
        # Collapse multiple spaces
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        return cleaned_text.strip()


class SummaryExtractor:
    """
    Cleans and extracts meeting summary text from local HTML files or raw markup.
    """

    def extract_summary_from_html(self, html_content: str) -> str:
        """
        Parses raw HTML string markup to extract cleaned plain text.

        Args:
            html_content: The HTML document markup.

        Returns:
            Cleaned plain text representation of the HTML document.
        """
        parser = SummaryHTMLParser()
        parser.feed(html_content)
        return parser.get_clean_text()

    def extract_summary_from_file(self, file_path: Path) -> str:
        """
        Loads an HTML file and extracts cleaned plain text content.

        Args:
            file_path: Path to the target HTML file.

        Returns:
            Cleaned plain text content.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"HTML summary file not found: {file_path}")

        html_content = file_path.read_text(encoding="utf-8", errors="ignore")
        return self.extract_summary_from_html(html_content)
