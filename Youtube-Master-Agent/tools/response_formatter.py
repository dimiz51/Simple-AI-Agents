from typing import List

"""
Tool used to format the response with the summaries, titles, topics, and URLs for all videos in a multi-line string

Args:
    summaries (List[str]): A list of summaries for the videos.
    titles (List[str]): A list of titles for the videos.
    topics (List[str]): A list of topics for the videos.
    urls (List[str]): A list of URLs for the videos.
Returns:
    str: A formatted string with the summaries, titles, topics, and URLs for all videos in a multi-line string.
"""


async def video_summary_response_formatter(
    summaries: List[str],
    titles: List[str],
    topics: List[str],
    urls: List[str],
) -> str:
    """Provides a formatted response with the summaries, titles, topics, URLs for all videos the user asked for."""
    # Ensure all lists have the same length
    if not (len(summaries) == len(titles) == len(topics) == len(urls)):
        return "Error: Input lists must have the same length."

    formatted_response = []

    for i in range(len(summaries)):
        formatted_response.append(f"Video {i+1}:")
        formatted_response.append(f"  Title: {titles[i]}")
        formatted_response.append(f"  URL: {urls[i]}")
        formatted_response.append(f"  Topic: {topics[i]}")
        formatted_response.append(f"  Summary: {summaries[i]}")
        formatted_response.append("")

    # Join all lines into a single string with line breaks
    return "\n".join(formatted_response)
