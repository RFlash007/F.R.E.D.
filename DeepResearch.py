import logging
import json
import ollama
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Global caches for research reports and queries history
research_cache = {}
queried_history = []

def scrape_web_data(query: str, max_results: int = 3) -> str:
    """
    Scrape websites for the given query using DuckDuckGo search.
    For each result, fetch the webpage and extract key text content.
    
    Args:
        query (str): The search query.
        max_results (int): Maximum number of websites to scrape.
        
    Returns:
        str: Combined extracted text content from the websites.
    """
    extracted_texts = []
    try:
        ddgs = DDGS()
        results = ddgs.text(keywords=query, region="us-en", safesearch="off", max_results=max_results)
        for res in results:
            url = res.get("href")
            if not url:
                continue
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    # Extract title and first 500 characters from paragraphs for brevity
                    title = soup.title.get_text().strip() if soup.title else "No Title"
                    paragraphs = soup.find_all("p")
                    page_text = " ".join([p.get_text().strip() for p in paragraphs])
                    snippet = page_text[:500].replace("\n", " ")
                    extracted_texts.append(f"URL: {url}\nTitle: {title}\nContent Snippet: {snippet}...\n")
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")
    except Exception as e:
        logging.error(f"Error performing DuckDuckGo search for query '{query}': {e}")
    return "\n\n".join(extracted_texts)

def perform_research(research_query: str, research_depth: int = 2) -> str:
    """
    Perform in-depth, multi-step research on a given topic using Ollama LLM.
    
    The final exhaustive report includes:
      - Executive Summary: Main findings and insights.
      - Detailed Methodology: Steps taken and research methods.
      - Systematic Analysis: Breakdown of key components.
      - Transparent Chain-of-Thought: Step-by-step reasoning.
      - Formatted Citations: Proper references.
      - Conclusions and Recommendations: Final insights.
      
    Additionally, the function keeps track of every query it searches for and,
    if a report already exists for a query, updates it with newly scraped information.
    It then recursively identifies related follow-up queries (the number corresponding
    to 'research_depth') and expands the research report with additional sections.
    
    Args:
        research_query (str): The research topic/query.
        research_depth (int): Depth for iterative research expansion.
        
    Returns:
        str: A comprehensive, exhaustive research report.
    """
    global research_cache, queried_history

    # Track the query
    if research_query not in queried_history:
        queried_history.append(research_query)
    
    # Check if a report already exists for this query;
    # if so, use it as the base to update further.
    if research_query in research_cache:
        report = research_cache[research_query]
    else:
        # Create an initial research prompt for the query.
        research_prompt = (
            f"Perform exhaustive, multi-step research on the topic: '{research_query}'. Include an Executive Summary, "
            "Detailed Methodology, Systematic Analysis, Transparent Chain-of-Thought, Formatted Citations, and "
            "Conclusions and Recommendations. Provide in-depth analysis and insights."
        )
        if research_depth > 1:
            research_prompt += f" Additionally, identify at least {research_depth} follow-up research queries that need further exploration. "
            research_prompt += f"This is research depth level {research_depth}."

        messages = [{"role": "user", "content": research_prompt}]
        try:
            response = ollama.chat(
                model="huihui_ai/qwen2.5-abliterate:14b",
                messages=messages,
                stream=False
            )
            report = response["message"]["content"]
        except Exception as e:
            logging.error(f"Error during primary research call for '{research_query}': {e}")
            report = f"Error during research: {e}"
    
    # Scrape websites related to the research query for additional data.
    scraped_data = scrape_web_data(research_query)
    if scraped_data:
        update_prompt = (
            f"Below is an existing research report on '{research_query}':\n\n{report}\n\n"
            f"Here is additional information scraped from relevant websites:\n\n{scraped_data}\n\n"
            "Please update and refine the research report with this additional information, ensuring citations are properly integrated."
        )
        messages = [{"role": "user", "content": update_prompt}]
        try:
            response_update = ollama.chat(
                model="huihui_ai/qwen2.5-abliterate:14b",
                messages=messages,
                stream=False
            )
            updated_report = response_update["message"]["content"]
            report = updated_report
        except Exception as e:
            logging.error(f"Error updating research report for '{research_query}': {e}")
    
    # Update the cache with the latest report.
    research_cache[research_query] = report

    # If deeper research is required, generate follow-up queries.
    if research_depth > 1:
        followup_prompt = (
            f"Based on the research on '{research_query}', identify {research_depth} follow-up research queries as a JSON array of strings."
        )
        messages = [{"role": "user", "content": followup_prompt}]
        try:
            response_followup = ollama.chat(
                model="huihui_ai/qwen2.5-abliterate:14b",
                messages=messages,
                stream=False
            )
            followup_content = response_followup["message"]["content"]
            followup_queries = json.loads(followup_content)
        except Exception as e:
            logging.error(f"Error retrieving or parsing follow-up queries for '{research_query}': {e}")
            followup_queries = []
        
        additional_reports = []
        for query in followup_queries:
            # Recursively perform research on each follow-up query,
            # decreasing the research depth by one.
            additional_report = perform_research(query, research_depth - 1)
            additional_reports.append(f"Follow-up on '{query}':\n{additional_report}")
        if additional_reports:
            report += "\n\n--- Additional Follow-up Research ---\n\n" + "\n\n".join(additional_reports)
        # Update the cache with combined follow-up research.
        research_cache[research_query] = report

    return report 

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test query
    test_query = "artificial intelligence ethics"
    research_depth = 2
    
    print(f"Performing research on: {test_query}")
    print(f"Research depth: {research_depth}")
    print("-" * 50)
    
    # Perform research
    report = perform_research(test_query, research_depth)
    
    # Print results
    print("\nResearch Report:")
    print("=" * 50)
    print(report)
    print("\nResearch Cache Status:")
    print(f"Cached queries: {list(research_cache.keys())}")
    print(f"Query history: {queried_history}")