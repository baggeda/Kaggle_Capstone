"""
Unit Tests for the Municipal Agenda Polling Agent
==================================================
Verifies RSS parsing, CivicClerk HTML scraping, state management,
and document downloading using mocks to avoid external network calls.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from ingestion.polling_agent import PollingAgent, PortalSource, AgendaItem


def test_rss_polling(tmp_path):
    """
    Verifies that standard RSS feeds are correctly parsed using XML libraries
    and mapped to AgendaItem schemas.
    """
    state_file = tmp_path / "state.json"
    download_dir = tmp_path / "downloads"
    agent = PollingAgent(state_file_path=state_file, download_dir=download_dir)

    mock_rss = """<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0">
        <channel>
            <title>Spartanburg Agenda Center</title>
            <item>
                <title>City Council Meeting - June 2026</title>
                <link>https://www.cityofspartanburg.org/AgendaCenter/ViewFile/Agenda/123</link>
                <pubDate>Mon, 22 Jun 2026 18:00:00 EST</pubDate>
            </item>
            <item>
                <title>Planning Commission - June 2026</title>
                <link>https://www.cityofspartanburg.org/AgendaCenter/ViewFile/Agenda/456</link>
                <pubDate>Tue, 23 Jun 2026 19:00:00 EST</pubDate>
            </item>
        </channel>
    </rss>
    """

    with patch.object(agent, 'fetch_url', return_value=mock_rss) as mock_fetch:
        source = PortalSource(name="Spartanburg", url="https://example.gov/rss", source_type="RSS")
        items = agent._poll_rss(source)

        assert len(items) == 2
        mock_fetch.assert_called_once_with("https://example.gov/rss")
        
        assert items[0].title == "City Council Meeting - June 2026"
        assert items[0].link == "https://www.cityofspartanburg.org/AgendaCenter/ViewFile/Agenda/123"
        assert items[0].published_date == "Mon, 22 Jun 2026 18:00:00 EST"
        assert items[0].source_name == "Spartanburg"

        assert items[1].title == "Planning Commission - June 2026"
        assert items[1].link == "https://www.cityofspartanburg.org/AgendaCenter/ViewFile/Agenda/456"


def test_civicclerk_polling(tmp_path):
    """
    Verifies that CivicClerk HTML landing pages are scraped using regex
    to extract PDF links and metadata.
    """
    state_file = tmp_path / "state.json"
    download_dir = tmp_path / "downloads"
    agent = PollingAgent(state_file_path=state_file, download_dir=download_dir)

    mock_html = """
    <html>
        <body>
            <div class="meetings">
                <a href="GenFile.aspx?ad=9876">June 2026 Agenda (PDF)</a>
                <a href="/Web/GenFile.aspx?meet=5432">June 2026 Agenda Packet</a>
                <a href="Player.aspx?id=111">Video and Agenda Portal</a>
                <a href="https://other.site/random.html">External Link</a>
            </div>
        </body>
    </html>
    """

    with patch.object(agent, 'fetch_url', return_value=mock_html) as mock_fetch:
        source = PortalSource(name="Greenville", url="https://greenvillesc.civicclerk.com/Web/Default.aspx", source_type="CivicClerk")
        items = agent._poll_civicclerk(source)

        # Should extract GenFile.aspx and Player.aspx links (3 matching links total)
        assert len(items) == 3
        
        # Absolute URL resolving checks
        assert items[0].link == "https://greenvillesc.civicclerk.com/Web/GenFile.aspx?ad=9876"
        assert "June 2026 Agenda (PDF)" in items[0].title

        assert items[1].link == "https://greenvillesc.civicclerk.com/Web/GenFile.aspx?meet=5432"
        assert "June 2026 Agenda Packet" in items[1].title

        assert items[2].link == "https://greenvillesc.civicclerk.com/Web/Player.aspx?id=111"
        assert "Video and Agenda Portal" in items[2].title


def test_state_filtering_and_saving(tmp_path):
    """
    Verifies state tracking to avoid reprocessing already scanned items.
    """
    state_file = tmp_path / "state.json"
    download_dir = tmp_path / "downloads"

    # Seed state file
    seed_state = {"processed_links": ["https://example.gov/agenda/123"]}
    state_file.write_text(json.dumps(seed_state), encoding="utf-8")

    agent = PollingAgent(state_file_path=state_file, download_dir=download_dir)

    # Check is_new
    assert not agent.is_new("https://example.gov/agenda/123")
    assert agent.is_new("https://example.gov/agenda/456")

    # Mark as processed
    agent.mark_processed("https://example.gov/agenda/456")
    assert not agent.is_new("https://example.gov/agenda/456")

    # Verify state written back to file correctly
    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "https://example.gov/agenda/456" in data["processed_links"]


def test_download_document(tmp_path):
    """
    Verifies that the file download successfully fetches content and
    writes it to the specified local path, updating state and item metadata.
    """
    state_file = tmp_path / "state.json"
    download_dir = tmp_path / "downloads"
    agent = PollingAgent(state_file_path=state_file, download_dir=download_dir)

    item = AgendaItem(
        title="June 2026 Meeting",
        link="https://greenvillesc.civicclerk.com/Web/GenFile.aspx?ad=123",
        source_name="Greenville"
    )

    # Mock response object for urllib.request.urlopen context manager
    mock_response = MagicMock()
    mock_response.read.return_value = b"%PDF-1.5 dummy PDF bytes"
    mock_response.__enter__.return_value = mock_response

    with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
        saved_path = agent.download_document(item)

        assert saved_path is not None
        assert Path(saved_path).exists()
        assert Path(saved_path).read_bytes() == b"%PDF-1.5 dummy PDF bytes"
        assert item.downloaded_path == saved_path
        
        # Verify URL is now processed
        assert not agent.is_new("https://greenvillesc.civicclerk.com/Web/GenFile.aspx?ad=123")
