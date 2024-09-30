import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加节点
G.add_node("前端模块", pos=(0, 2), description="用户交互和文件上传")
G.add_node("后端模块", pos=(2, 2), description="请求处理和数据管理")
G.add_node("模型侧服务模块", pos=(4, 2), description="核心匹配和检测")
G.add_node("目标检测子模块", pos=(4, 3), description="使用YOLOv5进行目标检测")
G.add_node("图像匹配与分类子模块", pos=(4, 1), description="双头输出模型")
G.add_node("知识蒸馏子模块", pos=(6, 2), description="教师-学生网络优化")

# 添加边
edges = [
    ("前端模块", "后端模块"),
    ("后端模块", "模型侧服务模块"),
    ("模型侧服务模块", "目标检测子模块"),
    ("模型侧服务模块", "图像匹配与分类子模块"),
    ("模型侧服务模块", "知识蒸馏子模块"),
]

G.add_edges_from(edges)

# 获取节点的位置
pos = nx.get_node_attributes(G, 'pos')

# 绘制图
font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=10000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True, arrowsize=20)
for node, (x, y) in pos.items():
    plt.text(x, y-0.1, G.nodes[node]['description'], fontsize=8, ha='center')
plt.title("系统整体架构图")
plt.savefig("patent_pic_1_系统整体架构图.png")
plt.show()
