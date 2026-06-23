"""
Municipal Agenda and RSS Polling Agent
======================================
Polls local government agenda portals and RSS feeds, prioritizing
South Carolina (Greenville, Greer, Spartanburg).
Uses standard libraries and Pydantic.
"""

import os
import re
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field


class PortalSource(BaseModel):
    """
    Configuration model for a municipal agenda portal source.
    """
    name: str = Field(..., description="Name of the municipality or board.")
    url: str = Field(..., description="The target URL for the feed or page.")
    source_type: str = Field("RSS", description="Type of source: 'RSS' or 'CivicClerk'.")


class AgendaItem(BaseModel):
    """
    Structured model representing a discovered legislative agenda item.
    """
    title: str = Field(..., description="Title of the agenda or meeting.")
    published_date: Optional[str] = Field(None, description="The date the agenda was published or discovered.")
    link: str = Field(..., description="URL to the online agenda details or PDF.")
    source_name: str = Field(..., description="Name of the source municipality.")
    downloaded_path: Optional[str] = Field(None, description="Local filepath where the PDF/text document is saved.")
    discovered_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of when the item was crawled."
    )


class PollingAgent:
    """
    Agent responsible for polling RSS feeds and CivicClerk portals,
    downloading new agendas, and maintaining tracking state.
    """

    def __init__(self, state_file_path: Path, download_dir: Path):
        """
        Initializes the PollingAgent with a tracking file and target download directory.
        """
        self.state_file_path = Path(state_file_path)
        self.download_dir = Path(download_dir)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # Create directories if they do not exist
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """
        Loads the history of processed links from the state file.
        """
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                # Fallback to empty state if parsing fails
                print(f"Warning: Failed to load state file ({e}). Starting fresh.")
        return {"processed_links": []}

    def _save_state(self) -> None:
        """
        Writes the current tracking history back to disk.
        """
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"Error saving state file: {e}")

    def is_new(self, link: str) -> bool:
        """
        Checks if a URL has already been processed by the agent.
        """
        return link not in self.state.get("processed_links", [])

    def mark_processed(self, link: str) -> None:
        """
        Adds a URL to the tracking history to prevent reprocessing.
        """
        if "processed_links" not in self.state:
            self.state["processed_links"] = []
        if link not in self.state["processed_links"]:
            self.state["processed_links"].append(link)
            self._save_state()

    def fetch_url(self, url: str) -> Optional[str]:
        """
        Helper method to perform GET requests with standard user agent headers.
        """
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                content_type = response.info().get_content_charset() or "utf-8"
                return response.read().decode(content_type, errors="ignore")
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return None

    def poll_all(self, sources: List[PortalSource]) -> List[AgendaItem]:
        """
        Polls a list of configured portal sources and retrieves any newly found agenda items.
        """
        new_items = []
        for src in sources:
            print(f"Polling source: {src.name} ({src.source_type})...")
            items = []
            if src.source_type.upper() == "RSS":
                items = self._poll_rss(src)
            elif src.source_type.upper() == "CIVICCLERK":
                items = self._poll_civicclerk(src)

            # Filter for new items
            for item in items:
                if self.is_new(item.link):
                    new_items.append(item)
        return new_items

    def _poll_rss(self, source: PortalSource) -> List[AgendaItem]:
        """
        Retrieves and parses RSS XML to identify agenda items.
        """
        xml_data = self.fetch_url(source.url)
        if not xml_data:
            return []

        items = []
        try:
            root = ET.fromstring(xml_data)
            # Support both RSS 2.0 and namespaces (e.g. Atom/DC)
            # Find all <item> tags
            for item_elem in root.findall(".//item"):
                title_elem = item_elem.find("title")
                link_elem = item_elem.find("link")
                pub_date_elem = item_elem.find("pubDate")

                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Untitled Meeting"
                link = link_elem.text.strip() if link_elem is not None and link_elem.text else None
                pub_date = pub_date_elem.text.strip() if pub_date_elem is not None and pub_date_elem.text else None

                if link:
                    items.append(
                        AgendaItem(
                            title=title,
                            published_date=pub_date,
                            link=link,
                            source_name=source.name
                        )
                    )
        except Exception as e:
            print(f"Error parsing RSS XML from {source.name}: {e}")
        return items

    def _poll_civicclerk(self, source: PortalSource) -> List[AgendaItem]:
        """
        Scrapes CivicClerk portal landing pages to find PDF agenda links.
        """
        html_data = self.fetch_url(source.url)
        if not html_data:
            return []

        items = []
        # Match links to PDF generation/downloads: GenFile.aspx?ad=..., GenFile.aspx?meet=..., or ending in .pdf
        # Also matches patterns like Player.aspx?id=... (which points to agenda/video combos)
        pattern = re.compile(r'href=["\'](?P<link>[^"\']*(?:GenFile\.aspx\?[^"\']+|Player\.aspx\?[^"\']+|\.pdf))["\']', re.IGNORECASE)
        matches = pattern.findall(html_data)

        # Regex to extract link text or surrounding text is complex,
        # so we also find anchors containing the text and matches.
        anchor_pattern = re.compile(r'<a\s+[^>]*href=["\'](?P<link>[^"\']*(?:GenFile\.aspx\?[^"\']+|Player\.aspx\?[^"\']+|\.pdf))["\'][^>]*>(?P<text>.*?)</a>', re.IGNORECASE | re.DOTALL)
        
        seen_links = set()
        for match in anchor_pattern.finditer(html_data):
            link = match.group("link").strip()
            # Clean HTML tags from text
            text = re.sub(r'<[^>]+>', '', match.group("text")).strip()
            
            # Form absolute URL
            absolute_link = urllib.parse.urljoin(source.url, link)

            # Prevent duplicates in the same run
            if absolute_link in seen_links:
                continue
            seen_links.add(absolute_link)

            # Clean up the parsed title
            title = text if text else "Legislative Document"
            # Limit length and remove excessive white spaces
            title = " ".join(title.split())
            if len(title) > 100:
                title = title[:97] + "..."

            items.append(
                AgendaItem(
                    title=f"{source.name} - {title}",
                    published_date=None,
                    link=absolute_link,
                    source_name=source.name
                )
            )

        # Fallback if anchor text parsing misses links but URLs were found
        for matched_link in matches:
            absolute_link = urllib.parse.urljoin(source.url, matched_link)
            if absolute_link not in seen_links:
                seen_links.add(absolute_link)
                items.append(
                    AgendaItem(
                        title=f"{source.name} - Legislative File ({matched_link.split('?')[-1][:20]})",
                        published_date=None,
                        link=absolute_link,
                        source_name=source.name
                    )
                )

        return items

    def download_document(self, agenda: AgendaItem) -> Optional[str]:
        """
        Downloads the PDF/document linked in the agenda item.

        Args:
            agenda: The AgendaItem target.

        Returns:
            The local file path where it was saved, or None if download failed.
        """
        url = agenda.link
        # Determine a filename based on title or URL
        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', agenda.source_name + "_" + agenda.title)
        filename = f"{safe_title[:60]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        local_path = self.download_dir / filename

        req = urllib.request.Request(url, headers=self.headers)
        try:
            print(f"Downloading: {url} to {local_path}")
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(local_path, "wb") as out_file:
                    out_file.write(response.read())
            agenda.downloaded_path = str(local_path)
            self.mark_processed(url)
            return str(local_path)
        except Exception as e:
            print(f"Failed to download document from {url}: {e}")
            return None


# Default SC prioritizations configuration list
DEFAULT_SC_SOURCES = [
    PortalSource(
        name="Spartanburg Agenda Center",
        url="https://www.cityofspartanburg.org/rss.aspx?type=agenda",
        source_type="RSS"
    ),
    PortalSource(
        name="Greenville City Council",
        url="https://greenvillesc.civicclerk.com/Web/Default.aspx",
        source_type="CivicClerk"
    ),
    PortalSource(
        name="Greer City Council",
        url="https://cityofgreersc.civicclerk.com/Web/Default.aspx",
        source_type="CivicClerk"
    )
]
