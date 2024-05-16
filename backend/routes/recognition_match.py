from fastapi import APIRouter, WebSocket
# from utils import pprint_prompt
from ws.constants import APP_ERROR_WEB_SOCKET_CODE  # type: ignore


router = APIRouter()

def write_result():
    pass

@router.websocket("/recognition")
async def recognition(websocket:WebSocket):
    pass  

@router.post("/recognition")
async def recognitionbypost():
    pass

