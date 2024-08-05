import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx

# 创建有向图
G = nx.DiGraph()

# 添加节点
G.add_node("教师网络", pos=(0, 1), description="复杂模型")
G.add_node("学生网络", pos=(2, 1), description="轻量模型")
G.add_node("输入数据", pos=(1, 0), description="训练数据")
G.add_node("特征蒸馏", pos=(1, 2), description="特征层蒸馏")
G.add_node("输出优化", pos=(3, 1), description="结果优化")

# 添加边
edges = [
    ("输入数据", "教师网络"),
    ("输入数据", "学生网络"),
    ("教师网络", "特征蒸馏"),
    ("特征蒸馏", "学生网络"),
    ("学生网络", "输出优化")
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
nx.draw(G, pos, with_labels=True, node_size=4000, node_color='lightgreen', font_size=10, font_weight='bold', arrows=True, arrowsize=20)
for node, (x, y) in pos.items():
    plt.text(x, y-0.1, G.nodes[node]['description'], fontsize=8, ha='center')
plt.title("教师-学生网络知识蒸馏训练策略示意图")
plt.savefig('patent_pic_4_教师-学生网络知识蒸馏训练策略示意图.png')
