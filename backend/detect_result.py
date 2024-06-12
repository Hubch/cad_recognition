import asyncio
from typing import List,Union,Awaitable, Callable

STREAM_CHUNK_SIZE = 2000


async def detect_completion(
    process_chunk: Callable[[str], Awaitable[None]], result: List[Union[str, None]]
) -> str:
    html = generate_html_response(result)
    
    for i in range(0, len(html), STREAM_CHUNK_SIZE):
        await process_chunk(html[i : i + STREAM_CHUNK_SIZE])
        await asyncio.sleep(0.01)
        
    return html


def generate_html_response(result):
    # Add heading
    html = f"<html>{HEAD}<body>"
    # Add detect result
    html += f"""
    <div class="container">
		<h3>元件识别结果</h3>
        <div class="iamge-container">
    		<img class="responsive-image" src="data:image/png;base64,{result[0]["detect_result"]["image_with_box"]}" >
    		<img class="responsive-image" src="data:image/png;base64,{result[1]["detect_result"]["image_with_box"]}">
        </div>
      <div>
        <div class="iamge-container">
            <h3>OCR转换结果1</h3>
    		<img class="responsive-image" src="data:image/png;base64,{result[0]["ocr_result"]["draw_image"]}">
            <h4>文本内容</h4>
            <div type="text" class="input">
                {",".join(result[0]["ocr_result"]["texts"])}
            </div>
            <h3>OCR转换结果2</h3>
    		<img class="responsive-image" src="data:image/png;base64,{result[1]["ocr_result"]["draw_image"]}">
            <h4>文本内容</h4>
            <div type="text" class="input">
                {",".join(result[1]["ocr_result"]["texts"])}
            </div>
        </div>
      </div>
    </div>
    """
    html += f"""<script>{JS}</script>"""
    return html


def generate_match_response(message):
    html = f"<div class='result'>{message}</div>"
    html += f"{MACTCH_REUSLT}"
    html += "</body></html>"
    return html


def generate_org_response(message):
      # Add heading
    html = f"<html>{HEAD}<body>"
    # imageHtmls =""
    # for image in message:
    #     imageHtmls += f'<img class="responsive-image" src="data:image/png;base64,{image}">'
    # html += f"""
    #   <div class="iamge-container">
    #     {imageHtmls}
    #   </div>
    # """
    return html

def generate_image_response(message,detail):
    # Add heading
    #html = f"<html>{HEAD}<body>"
    html=""
    if detail:
        plist = "<p>" + "</p><p>".join(detail) + "</p>"
    else:
        plist ="没有差异"
    imageHtmls =""
    for image in message:
        imageHtmls += f'<img class="responsive-image" src="data:image/png;base64,{image}">'
   
    html +=f"""
    <h3>对比结果</h3>
    <div class="iamge-container">
        {imageHtmls}
    </div>
    <h3>详细区别</h3>
    <div class ="input">{plist}</div>
  
    """
    # html += f"""
    #   <h3>总结</h3>
    # <div class ="input" >
    # {describe}
    # </div>
    # """
    html += f"""<script>{JS}</script>"""
    return html

def llm_emp_response():
    html = f"<html>{HEAD}<body>"
    html +=f"""
    <h3>对比失败,网络异常，大模型无响应</h3>
    """
    html += f"""<script>{JS}</script>"""
    return html
    
    
# class ResultInfo:
#     org_images:list[str]
#     detect_image:list[str]
#     ocr_result =list[str]
#     ocr_image =list[str]
CLASS = """
.container {
	flex-direction: column;
	align-items: center;
    margin-top: 50px;
}

.image {
	margin: 20px;
	border: 1px solid #ddd;
	box-shadow: 0 0 5px #ddd;
	max-width: 500px;
	max-height: 500px;
}

.result {
    width:100%
}

.iamge-container {
    display:inline
}

.input {
    color: #565772;
    border: 1px solid #D9DEE9;
    border-radius: 6px;
    padding: 12px;
    line-height: 24px;
    min-height: 72px;
    max-height: 400px;
    overflow: auto;
    width:100%
}

pre {
    background-color: #272822;
    color: #f8f8f2;
    padding: 5px;
    border-radius: 5px;
    overflow-x: auto;
}

pre code {
      display: block;
      padding: 1em;
      overflow-x: auto;
      background-color: #f4f4f4;
      border: 1px solid #ddd;
    }
.responsive-image {
    width: 100%;
    max-width: 500px; /* 初始图片的最大宽度 */
    cursor: pointer; /* 鼠标悬停时显示指针，表示可点击 */
    transition: transform 0.3s ease; /* 平滑缩放效果 */
}
        
.responsive-image.expanded {
    transform: scale(3); /* 放大图片 */
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(3);
    cursor: grab;
    z-index: 10;
}
"""
JS ="""
    const images = document.querySelectorAll('.responsive-image');
    images.forEach(img => {
        img.addEventListener('click', function() {
            // 切换图片的类来放大或恢复
            this.classList.toggle('expanded');
        });
    });
"""
   
HEAD = f"""
<head>
	<meta charset="UTF-8">
	<title>检测结果</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/styles/default.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/highlight.min.js"></script>
    <script>hljs.initHighlightingOnLoad();</script>
	<style>
		{CLASS}
	</style>
</head>
"""

MACTCH_REUSLT ="""
	<div class="result">
      <pre>
        <code>
          <p style="display:inline">对比结果：<span style="color:green">完全一致</span>
          </p>
          <p style="display:inline">运行耗时：<span style="color:green">100ms</span>
          </p>
        </code>
      </pre>
	</div>
"""

DETECT_RESULT =f"""
<div class="container">
		<h3>元件识别结果</h3>
        <div class="iamge-container">
    		<img class="responsive-image" src="data:image/png;base64," >
    		<img class="responsive-image" src="'data:image/png;base64,">
        </div>
      <div>
        <div class="iamge-container">
            <h3>OCR转换结果1</h3>
    		<img class="responsive-image" src="/outputs/image.png">
            <h4>文本内容</h4>
            <div type="text" class="input">
              减液界区闵组附近 综合 W-530 B-73 备及 IN 7307 50-SW-3001-A1X 186D-01140-004 40-FW-3021-A1A- FIRO SL FIRO 304 SL 31 SL ON2 HoNa 23 50-IA-3001-A1X-N X 80-PA-3001-A1A-N X 236 化 7 SL N1 至脱水机SPR-7501A/ 1309 PD-01140-007 去至絮凝剂进料泵吹 X 文 仅表伴热用 80 X 80-HWS-3001-A1A-H X X 232 SL SI < 23 X 7302布置在立管上，N1AF与N2AF之问距 310 X 仅表伴热用 又 X 氧化风机冷却月 氧化风机冷却月 氧化风机冷却用 浊度仪冷却月 T-1502/AT-7503 8 K-1501A K-1501B K-15010 X X 273 10-CWS-3001-A1A. X HoeNd < 200-LFW-3001-A1A- X 文 X 50-JFW-3004-A1A-1 # # N-VI4-800 731 11 T305 来自TICSA-730 227PD-01140-00 730 IXI 120 7302 303 PID.01 修改标 日期 至除小激冷塔r- X 领方式扩做至第三 中石化字药工程有限公司 隆温晴电 1B-A1 装置或单元（ 400. 5G-30014 00x2 设计阶段 专业名称 次 HE
            </div>
            <h3>OCR转换结果2</h3>
    		<img class="responsive-image" src="/outputs/image.png">
            <h4>文本内容</h4>
            <div type="text" class="input">
              减液界区闵组附近 综合 W-530 B-73 备及 IN 7307 50-SW-3001-A1X 186D-01140-004 40-FW-3021-A1A- FIRO SL FIRO 304 SL 31 SL ON2 HoNa 23 50-IA-3001-A1X-N X 80-PA-3001-A1A-N X 236 化 7 SL N1 至脱水机SPR-7501A/ 1309 PD-01140-007 去至絮凝剂进料泵吹 X 文 仅表伴热用 80 X 80-HWS-3001-A1A-H X X 232 SL SI < 23 X 7302布置在立管上，N1AF与N2AF之问距 310 X 仅表伴热用 又 X 氧化风机冷却月 氧化风机冷却月 氧化风机冷却用 浊度仪冷却月 T-1502/AT-7503 8 K-1501A K-1501B K-15010 X X 273 10-CWS-3001-A1A. X HoeNd < 200-LFW-3001-A1A- X 文 X 50-JFW-3004-A1A-1 # # N-VI4-800 731 11 T305 来自TICSA-730 227PD-01140-00 730 IXI 120 7302 303 PID.01 修改标 日期 至除小激冷塔r- X 领方式扩做至第三 中石化字药工程有限公司 隆温晴电 1B-A1 装置或单元（ 400. 5G-30014 00x2 设计阶段 专业名称 次 HE
            </div>
        </div>
      </div>
</div>
"""

RESULT = f"""
<html>
{HEAD}
<body>
	{DETECT_RESULT}
    {MACTCH_REUSLT}
<script>
  {JS}
</script>
</body>
</html>
"""