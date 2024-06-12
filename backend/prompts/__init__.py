from typing import List, NoReturn, Union

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionContentPartParam

from prompts.imported_code_prompts import IMPORTED_CODE_SYSTEM_PROMPTS
from prompts.screenshot_system_prompts import SYSTEM_PROMPTS
from prompts.match_cad_prompts import CAD_MATCH_SYSTEM_PROMPTS
from prompts.types import Stack
import json


USER_PROMPT = """
Generate code for a web page that looks exactly like this.
"""

SVG_USER_PROMPT = """
Generate code for a SVG that looks exactly like this.
"""


def assemble_imported_code_prompt(
    code: str, stack: Stack, result_image_data_url: Union[str, None] = None
) -> List[ChatCompletionMessageParam]:
    system_content = IMPORTED_CODE_SYSTEM_PROMPTS[stack]

    user_content = (
        "Here is the code of the app: " + code
        if stack != "svg"
        else "Here is the code of the SVG: " + code
    )
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    # TODO: Use result_image_data_url


def assemble_prompt(
    image_data_url: str,
    stack: Stack,
    result_image_data_url: Union[str, None] = None,
) -> List[ChatCompletionMessageParam]:
    system_content = SYSTEM_PROMPTS[stack]
    user_prompt = USER_PROMPT if stack != "svg" else SVG_USER_PROMPT

    user_content: List[ChatCompletionContentPartParam] = [
        {
            "type": "image_url",
            "image_url": {"url": image_data_url, "detail": "high"},
        },
        {
            "type": "text",
            "text": user_prompt,
        },
    ]

    # Include the result image if it exists
    if result_image_data_url:
        user_content.insert(
            1,
            {
                "type": "image_url",
                "image_url": {"url": result_image_data_url, "detail": "high"},
            },
        )
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]




JSON_USER_PROMPT = """
你是一个CAD审图专家，可以通过对比两个结构化元件信息找出元件和文本的差异。
"""

def assemble_json_prompt(
    detection_info: List[Union[str, None]],
    stack: Stack,
) -> List[ChatCompletionMessageParam]:
    system_content = CAD_MATCH_SYSTEM_PROMPTS.get(stack, "Default System Prompt")
    user_prompt = JSON_USER_PROMPT

    user_content: List[ChatCompletionContentPartParam] = [
        {
            "type": "text",
            "text": user_prompt,
        },
    ]

    # Include detection_info as text parts in user_content
    for info in detection_info:
        if info:  # Make sure the info is not None or empty
            user_content.append(
                {
                    "type": "text",
                    "text": json.dumps(info),
                }
            )
            
    # Construct the final list of ChatCompletionMessageParam objects
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

