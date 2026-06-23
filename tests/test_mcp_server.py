"""
Unit Tests for the Municipal Agenda MCP Server
==============================================
Verifies the tools exposed by the MCP server using mocks to isolate
the server's interface from active web queries.
"""

from unittest.mock import patch, MagicMock
from ingestion.polling_agent import AgendaItem
from mcp_server.server import list_upcoming_meetings, index_meeting, get_indexed_meetings


@patch('mcp_server.server.agent')
def test_list_upcoming_meetings_tool(mock_agent):
    """
    Verifies that list_upcoming_meetings tool correctly polls and formats results.
    """
    # Configure mock polling agent return items
    mock_item = AgendaItem(
        title="Spartanburg - City Council Meeting",
        link="https://www.cityofspartanburg.org/AgendaCenter/ViewFile/Agenda/123",
        source_name="Spartanburg"
    )
    mock_agent.poll_all.return_value = [mock_item]

    # Run the tool
    results = list_upcoming_meetings(city="spartanburg")

    # Assertions
    mock_agent.poll_all.assert_called_once()
    assert len(results) == 1
    assert results[0]["title"] == "Spartanburg - City Council Meeting"
    assert results[0]["link"] == "https://www.cityofspartanburg.org/AgendaCenter/ViewFile/Agenda/123"


@patch('mcp_server.server.agent')
def test_index_meeting_tool(mock_agent):
    """
    Verifies that the index_meeting tool downloads and reports status correctly.
    """
    # 1. Test successful download/index
    mock_agent.is_new.return_value = True
    mock_agent.download_document.return_value = "data/raw/Greenville_June_Meeting.pdf"

    result = index_meeting(
        meeting_title="June Meeting",
        meeting_link="https://greenvillesc.civicclerk.com/Web/GenFile.aspx?ad=123",
        source_name="Greenville"
    )

    assert result["status"] == "success"
    assert result["downloaded_path"] == "data/raw/Greenville_June_Meeting.pdf"
    mock_agent.download_document.assert_called_once()

    # 2. Test already indexed item
    mock_agent.is_new.return_value = False
    result_already = index_meeting(
        meeting_title="June Meeting",
        meeting_link="https://greenvillesc.civicclerk.com/Web/GenFile.aspx?ad=123",
        source_name="Greenville"
    )
    assert result_already["status"] == "already_indexed"
    assert result_already["downloaded_path"] is None


@patch('mcp_server.server.agent')
def test_get_indexed_meetings_tool(mock_agent):
    """
    Verifies that the get_indexed_meetings tool lists current processed URLs.
    """
    mock_agent.state = {"processed_links": ["https://example.gov/agenda/123"]}

    results = get_indexed_meetings()

    assert len(results) == 1
    assert results[0] == "https://example.gov/agenda/123"
