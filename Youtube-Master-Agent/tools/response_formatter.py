from typing import List
import json


async def json_response_formatter(
    summaries: List[str],
    titles: List[str],
    topics: List[str],
    urls: List[str],
) -> str:
    """Provides a JSON-formatted response with the summaries, titles, topics, URLs for all videos the user asked for."""
    # Ensure all lists have the same length
    if not (len(summaries) == len(titles) == len(topics) == len(urls)):
        return "Error: Input lists must have the same length."

    formatted_response = []

    formatted_response = [
        {
            "video_title": titles[i],
            "url": urls[i],
            "topic": topics[i],
            "summary": summaries[i],
        }
        for i in range(len(summaries))
    ]

    # Convert the list to a JSON string
    return json.dumps({"Youtube Videos": formatted_response}, indent=2)
