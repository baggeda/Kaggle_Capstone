"""
MCP Server for Municipal Agenda Portals
=======================================
Exposes tools to interface with city APIs and index meeting agendas
via Model Context Protocol (MCP).
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from contextlib import redirect_stdout
from mcp.server.fastmcp import FastMCP

from ingestion.polling_agent import PollingAgent, PortalSource, AgendaItem, DEFAULT_SC_SOURCES

# Initialize the server name
mcp = FastMCP("Municipal Agenda Indexer")

# Resolve directories relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
STATE_FILE = PROJECT_ROOT / "data" / "polling_state.json"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"

# Create agent instance
agent = PollingAgent(state_file_path=STATE_FILE, download_dir=DOWNLOAD_DIR)


@mcp.tool()
def list_upcoming_meetings(city: str = "all") -> List[Dict[str, Any]]:
    """
    Polls target city portals and lists upcoming meetings.

    Args:
        city: The name of the city to filter by (e.g. 'greenville', 'greer', 'spartanburg', or 'all').

    Returns:
        A list of dictionary items representing discovered upcoming meetings.
    """
    # Defensive redirection of stdout to stderr to avoid corrupting the stdio JSON-RPC stream
    with redirect_stdout(sys.stderr):
        sources = DEFAULT_SC_SOURCES
        city_lower = city.lower().strip()

        if city_lower != "all":
            # Filter sources that match the target city name
            sources = [s for s in sources if city_lower in s.name.lower()]

        if not sources:
            return []

        # Poll all matching sources
        items = agent.poll_all(sources)
        
        # Serialize list of Pydantic AgendaItem models to standard dict format
        return [item.model_dump() for item in items]


@mcp.tool()
def index_meeting(meeting_title: str, meeting_link: str, source_name: str) -> Dict[str, Any]:
    """
    Downloads and indexes a specific municipal meeting agenda document.

    Args:
        meeting_title: The title or description of the meeting.
        meeting_link: The URL to the meeting's agenda PDF/file.
        source_name: The name of the city or municipality (e.g., 'Greenville').

    Returns:
        A dictionary with the indexing status and the local file path.
    """
    with redirect_stdout(sys.stderr):
        item = AgendaItem(
            title=meeting_title,
            link=meeting_link,
            source_name=source_name
        )
        
        # Check if already processed
        if not agent.is_new(meeting_link):
            return {
                "status": "already_indexed",
                "message": f"Meeting at {meeting_link} is already indexed.",
                "downloaded_path": None
            }

        # Download the document
        local_path = agent.download_document(item)
        
        if local_path:
            return {
                "status": "success",
                "message": f"Successfully downloaded and indexed: {meeting_title}",
                "downloaded_path": local_path
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to download agenda from {meeting_link}",
                "downloaded_path": None
            }


@mcp.tool()
def get_indexed_meetings() -> List[str]:
    """
    Retrieves the list of URLs of all meetings that have already been indexed/downloaded.

    Returns:
        A list of meeting URLs.
    """
    return agent.state.get("processed_links", [])


if __name__ == "__main__":
    # Start FastMCP server on standard I/O (default)
    mcp.run(transport="stdio")
