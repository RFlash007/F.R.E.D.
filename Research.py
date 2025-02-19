import json
from duckduckgo_search import DDGS
import ollama
from pydantic import BaseModel
from Semantic import update_semantic
import Notes
import logging

class ResearchAnalysis(BaseModel):
    """
    Represents simplified research analysis
    """
    summary: str
    key_points: list[str]
    new_directions: list[str]

    def to_json(self) -> str:
        """Convert analysis to JSON format"""
        return json.dumps(self.model_dump())

    @classmethod
    def from_json(cls, json_str: str) -> "ResearchAnalysis":
        """Create analysis from JSON string"""
        return cls.model_validate_json(json_str)

class ResearchAgent:
    def __init__(self, max_depth=2):
        self.ddg = DDGS()
        self.max_depth = max_depth
        self.paper = ""
        self.name = ""

    def research(self, topic, depth=0):
        """Autonomous research with depth control"""
        if depth >= self.max_depth:
            return ""
        if depth == 0:
            self.name = topic
        # 1. Multi-source search
        results = self._search(topic)
        if not results:
            return "No relevant sources found"
            
        # 2. AI analysis
        analysis_prompt = f"""Analyze these sources about {topic}:
        {results}
        Return a JSON object with this structure:
        {{
            "summary": "Overall summary of findings",
            "key_points": ["point1", "point2"...],
            "new_directions": ["direction1", "direction2"...]
        }}"""

        response = ollama.chat(model="FRED", messages=[{
            "role": "user",
            "content": analysis_prompt
        }])

        try:
            analysis = ResearchAnalysis.from_json(response['message']['content'])
            # 3. Update memory
            # 2. AI analysis
            analysis_prompt = f"""Analyze this summary about: {topic}:
            Summary: {analysis.summary}
            Key Points: {analysis.key_points}
            Based on this summary, edit this summary, incorporating all new information whilst keeping the original information (if any) intact: {self.paper}
            Return the edited summary.
            ###RETURN ONLY THE SUMMARY, NO OTHER TEXT###
            """

            response = ollama.chat(model="FRED", messages=[{
                "role": "user",
                "content": analysis_prompt
            }])
            self.paper = response['message']['content']
            # 4. Explore new directions
            for direction in analysis.new_directions[:2]:
                self.research(direction, depth+1)
                
            return Notes.create_note(self.name, self.paper)
        except Exception as e:
            print(f"Error parsing analysis: {str(e)}")  # Added error printing
            return "Error parsing analysis"

    def _search(self, query):
        """
        Perform a DuckDuckGo-based search for research topics.

        Args:
            query (str): The research topic to search for.

        Returns:
            str: Combined search results from text and news searches.
        """
        region = "us-en"
        safesearch = "off"

        # Gather results
        text_results = list(self.ddg.text(
            keywords=query,
            region=region,
            safesearch=safesearch,
            max_results=2
        ))
        news_results = list(self.ddg.news(
            keywords=query,
            region=region,
            safesearch=safesearch,
            max_results=2
        ))

        # Combine results into single string
        combined_results = (
            f"[Text Results]\n{text_results}\n\n"
            f"[News Results]\n{news_results}"
        )

        return combined_results

    def _get_directions(self, analysis):
        """Extract research directions from analysis JSON"""
        try:
            analysis_obj = ResearchAnalysis.from_json(analysis)
            return analysis_obj.new_directions
        except Exception:
            return []

if __name__ == "__main__":
    Notes.delete_note("AI")
