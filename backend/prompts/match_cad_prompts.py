from prompts.types import SystemPrompts

RETURN_FORMAT="""
{
  "commonalities": {
    "objects": ["List", "of", "common", "objects"],
    "scenes": ["Similar", "scenes", "or", "settings"],
    "colors": ["Common", "color", "schemes", "or", "tones"]
  },
  "differences": {
    "objects": ["List", "of", "unique", "objects", "in", "each", "image"],
    "actions": ["Different", "actions", "performed", "by", "people", "or", "objects"],
    "textures": ["Distinct", "textures", "or", "patterns"],
    "lighting": ["Variations", "in", "light", "source", "or", "intensity"]
  }
}
"""
ASPECT = """
location, category
"""

JSON_SYSTEM_PROMPT =f"""
You are an expert in comparing picture information.
You have now got the image recognition information and ocr information of two images from the user, and then you need to judge whether the analysis is consistent or not based on this information, and output the inconsistent parts as per the format.

- Analyze the following two images and return the similarities and differences in a structured JSON format.
- Ensure the analysis includes but is not limited to the following aspects:{ASPECT}. 
- Provide detailed descriptions and comparisons for each aspect.

Return in full in the following format

Example of the return format:
{RETURN_FORMAT}
"""

CAD_MATCH_SYSTEM_PROMPTS = SystemPrompts(
    html_tailwind="",
    react_tailwind="",
    bootstrap="",
    ionic_tailwind="",
    vue_tailwind="",
    svg="",
    json=JSON_SYSTEM_PROMPT,
)
