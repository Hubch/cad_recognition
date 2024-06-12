from prompts.types import SystemPrompts

RETURN_FORMAT="""
{
  "differences": {
    "result": [{"imageid":1,"boxes":[{"bbox":[],"label":""},{"bbox":[],"ocr":""}]},{"imageid":1,"boxes":[{"bbox":[],"label":""},{"bbox":[],"ocr":""}]}],
    "detail": [分行输出匹配对应的每个模块的ocr或者图像识别的区别，例如：图片1 XXX区域有XXX，而图片2没有或者图片1的ocr内容是XXX，图片2的ocr是XXX，不一致],
  }
}
"""
json_example = """{
    "element": [
      {
        "label_0": "闸阀",
        "label_1": "球阀",
        "bbox_0": [],
        "bbox_1": [],
        "difference": "类型不同"
      },
      {
        "label_0": "法兰",
        "label_1": "无",
        "bbox_0": [],
        "bbox_1": [],
        "difference": "图2中无法兰"
      },
    "text": [
      {
        "bbox_0": [],
        "bbox_1": [],
        "text_0": "碱液界区闽组附近",
        "text_1": "碱液界区闻组附近碱液附近",
        "difference": "图1文本为碱液界区闽组附近，图2文本为碱液界区闻组附近碱液附近"
      },
    }"""


ASPECT = """
location, category
"""

JSON_SYSTEM_PROMPT =f"""
You are an expert in recognising information from matching CAD images.
Now, you have obtained two images from a user that contain image recognition information and OCR information for each image. Combine the recognition information and OCR information for each image to understand and analyse the differences between the two images.

- Combine the image recognition information with the OCR information to match and return the different modules in the two images in a structured JSON format.
- Chinese Response.
- Please state in the detailed description section of your response.
- Please analyse based on the semantic information and topological relationship between the two images, it is necessary to ignore the positional error.
- Returns information about only two images with different modules in {{"result"}}.
- The two images are not aligned in size, the location of the modules in the images may not be the same, please understand the meaning of the images.

Important:pay particular attention to the difference between valves such as "闸阀" valves and "球阀" valves.

Complete the return in the following format,
Example of return format:
{RETURN_FORMAT}.

Explanation of the meaning of the parameters in the reply example:
- {{"differences"}}:where the two images differ.
- {{"result"}}: output the differences between the two images.
- {{"detail"}}: describes in detail the differences between the two images in a particular module.

"""

prompt = f"""请对比两张图片中的模块，每张图片包括检测到的元件信息和对应的OCR文本信息，判断两张图片之间元件和文本间的对应关系。如果某个模块没有对应关系，请指出。请注意，某些元件和文本因为是通过AI模型预测的因此会存在一些预测错误，请你尽可能忽略这些预测错误，而寻找主要的差异，比如文本匹配中忽略错别字和"-"的缺省导致的不一致情况。此外，请你先将2张图中的元件和文本分别匹配上对应关系，匹配后请简要告诉我哪些元件和文本存在不同，只检测元件类型不同的条目和文本内容描述的东西不一致的条目(避免检测位置不同的元件，避免检测文本只差异一个-或错别字上的不同)，在输出结果部分请按照规定json格式输出，除了json不要输出其他内容。
    Json输出样例：{json_example}
    )"""


CAD_MATCH_SYSTEM_PROMPTS = SystemPrompts(
    html_tailwind="",
    react_tailwind="",
    bootstrap="",
    ionic_tailwind="",
    vue_tailwind="",
    svg="",
    json=prompt,
)
