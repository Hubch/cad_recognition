import os
from fastapi import APIRouter, WebSocket
import openai
from config import ANTHROPIC_API_KEY, IS_PROD, SHOULD_MOCK_AI_RESPONSE
from custom_types import InputMode
from llm import (
    Llm,
    convert_frontend_str_to_llm,
    stream_claude_response,
    stream_openai_response,
)
from openai.types.chat import ChatCompletionMessageParam
from mock_llm import mock_completion
from typing import Dict, List, cast, get_args
from prompts import assemble_json_prompt
from datetime import datetime
import json
from prompts.types import Stack

# from utils import pprint_prompt
from ws.constants import APP_ERROR_WEB_SOCKET_CODE  # type: ignore
from pydantic import BaseModel
import httpx


router = APIRouter()


def write_logs(prompt_messages: List[ChatCompletionMessageParam], completion: str):
    # Get the logs path from environment, default to the current working directory
    logs_path = os.environ.get("LOGS_PATH", os.getcwd())

    # Create run_logs directory if it doesn't exist within the specified logs path
    logs_directory = os.path.join(logs_path, "run_logs")
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    print("Writing to logs directory:", logs_directory)

    # Generate a unique filename using the current timestamp within the logs directory
    filename = datetime.now().strftime(f"{logs_directory}/messages_%Y%m%d_%H%M%S.json")

    # Write the messages dict into a new file for each run
    with open(filename, "w") as f:
        f.write(json.dumps({"prompt": prompt_messages, "completion": completion}))


@router.websocket("/recognition")
async def recognition(websocket:WebSocket):
    await websocket.accept()

    print("Incoming websocket connection...")

    async def throw_error(
        message: str,
    ):
        await websocket.send_json({"type": "error", "value": message})
        await websocket.close(APP_ERROR_WEB_SOCKET_CODE)

    # TODO: Are the values always strings?
    params: Dict[str, str] = await websocket.receive_json()

    print("Received params")
    # Read the code config settings from the request. Fall back to default if not provided.
    generated_code_config = ""
    if "generatedCodeConfig" in params and params["generatedCodeConfig"]:
        generated_code_config = params["generatedCodeConfig"]
    if not generated_code_config in get_args(Stack):
        await throw_error(f"Invalid generated code config: {generated_code_config}")
        return
    
    # Validate the input mode
    input_mode = params.get("inputMode")
    if not input_mode in get_args(InputMode):
        await throw_error(f"Invalid input mode: {input_mode}")
        raise Exception(f"Invalid input mode: {input_mode}")
    # Cast the variable to the right type
    validated_input_mode = cast(InputMode, input_mode)

    # Read the model from the request. Fall back to default if not provided.
    code_generation_model_str = params.get(
        "codeGenerationModel", Llm.GPT_4O_2024_05_13.value
    )
    try:
        code_generation_model = convert_frontend_str_to_llm(code_generation_model_str)
    except:
        await throw_error(f"Invalid model: {code_generation_model_str}")
        raise Exception(f"Invalid model: {code_generation_model_str}")
    exact_llm_version = None

    print(
        f"Generating {generated_code_config} code for uploaded {input_mode} using {code_generation_model} model..."
    )

    # Get the OpenAI API key from the request. Fall back to environment variable if not provided.
    # If neither is provided, we throw an error.
    openai_api_key = None
    if params["openAiApiKey"]:
        openai_api_key = params["openAiApiKey"]
        print("Using OpenAI API key from client-side settings dialog")
    else:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            print("Using OpenAI API key from environment variable")

    if not openai_api_key and (
        code_generation_model == Llm.GPT_4_VISION
        or code_generation_model == Llm.GPT_4_TURBO_2024_04_09
        or code_generation_model == Llm.GPT_4O_2024_05_13
    ):
        print("OpenAI API key not found")
        await throw_error(
            "No OpenAI API key found. Please add your API key in the settings dialog or add it to backend/.env file. If you add it to .env, make sure to restart the backend server."
        )
        return

    # Get the OpenAI Base URL from the request. Fall back to environment variable if not provided.
    openai_base_url = None
    # Disable user-specified OpenAI Base URL in prod
    if not os.environ.get("IS_PROD"):
        if "openAiBaseURL" in params and params["openAiBaseURL"]:
            openai_base_url = params["openAiBaseURL"]
            print("Using OpenAI Base URL from client-side settings dialog")
        else:
            openai_base_url = os.environ.get("OPENAI_BASE_URL")
            if openai_base_url:
                print("Using OpenAI Base URL from environment variable")

    if not openai_base_url:
        print("Using official OpenAI URL")

    print("Recognizing cad...")
    await websocket.send_json({"type": "status", "value": "Recognizing cad..."})

    async def process_chunk(content: str):
        await websocket.send_json({"type": "chunk", "value": content})

    # pprint_prompt(prompt_messages)  # type: ignore
    detect_results = []
    
    for image in params["images"]:
        request = DetectRequest(image=image)
        detect_result = await modelserve(request)
        print(f"detect_result: {detect_result}")
        detect_results.append(detect_result)
    
    # Assemble the prompt
    try:
        prompt_messages = assemble_json_prompt(detect_result, "json")
    except:
        await websocket.send_json(
            {
                    "type": "error",
                    "value": "Error assembling prompt.",
            }
        )
        await websocket.close()
        return
    
    if SHOULD_MOCK_AI_RESPONSE:
        print("SHOULD_MOCK_AI_RESPONSE:",SHOULD_MOCK_AI_RESPONSE)
        completion = await mock_completion(
            process_chunk, input_mode=validated_input_mode
        )
    else:
        try:
            if code_generation_model == Llm.CLAUDE_3_SONNET:
                if not ANTHROPIC_API_KEY:
                    await throw_error(
                        "No Anthropic API key found. Please add the environment variable ANTHROPIC_API_KEY to backend/.env"
                    )
                    raise Exception("No Anthropic key")

                completion = await stream_claude_response(
                    prompt_messages,  # type: ignore
                    api_key=ANTHROPIC_API_KEY,
                    callback=lambda x: process_chunk(x),
                )
                exact_llm_version = code_generation_model
            else:
                completion = await stream_openai_response(
                    prompt_messages,  # type: ignore
                    api_key=openai_api_key,
                    base_url=openai_base_url,
                    callback=lambda x: process_chunk(x),
                    model=code_generation_model,
                )
                exact_llm_version = code_generation_model
       
        except openai.AuthenticationError as e:
            print("[GENERATE_CODE] Authentication failed", e)
            error_message = (
                "Incorrect OpenAI key. Please make sure your OpenAI API key is correct, or create a new OpenAI API key on your OpenAI dashboard."
                + (
                    " Alternatively, you can purchase code generation credits directly on this website."
                    if IS_PROD
                    else ""
                )
            )
            return await throw_error(error_message)
        except openai.NotFoundError as e:
            print("[GENERATE_CODE] Model not found", e)
            error_message = (
                e.message
                + ". Please make sure you have followed the instructions correctly to obtain an OpenAI key"
                + (
                    " Alternatively, you can purchase code generation credits directly on this website."
                    if IS_PROD
                    else ""
                )
            )
            return await throw_error(error_message)
        except openai.RateLimitError as e:
            print("[GENERATE_CODE] Rate limit exceeded", e)
            error_message = (
                "OpenAI error - 'You exceeded your current quota, please check your plan and billing details.'"
                + (
                    " Alternatively, you can purchase code generation credits directly on this website."
                    if IS_PROD
                    else ""
                )
            )
            return await throw_error(error_message)

    print("Exact used model for generation: ", exact_llm_version)

    # Write the messages dict into a log so that we can debug later
    write_logs(prompt_messages, completion)
    await websocket.close() 



class DetectResponse(BaseModel):
    org_image :str
    object_detect : str
    ocr_result : str

class DetectRequest(BaseModel):
    image:str


async def modelserve(request: DetectRequest):
    api_base_url = os.environ.get("MODEL_SERVE_URL", "http://localhost:8000/caddetect")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                api_base_url,
                files={"image": request.image},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                response.raise_for_status()
    except httpx.RequestError as e:
        raise Exception("Error taking model serve") from e
