import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# 设置中文字体
font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# 绘制流程图
def draw_process_flow():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制流程步骤框
    process_steps = [
        ("1. 超大图获取", (0.1, 0.8)),
        ("2. 双图像尺寸对齐", (0.1, 0.6)),
        ("3. 滑窗检测", (0.1, 0.4)),
        ("4. 全局坐标恢复与NMS", (0.1, 0.2)),
        ("5. 位置匹配（匈牙利算法）", (0.5, 0.6)),
        ("6. 元件匹配（余弦距离）", (0.5, 0.4)),
        ("7. 相似度与差异点分析\n导出为JSON", (0.8, 0.4))
    ]
    
    # 绘制矩形框和文字
    for step, (x, y) in process_steps:
        ax.add_patch(patches.Rectangle((x, y), 0.3, 0.1, edgecolor='black', facecolor='lightblue', lw=2))
        ax.text(x + 0.15, y + 0.05, step, ha='center', va='center', fontproperties=font_prop)
    
    # 绘制箭头
    arrow_params = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate("", xy=(0.25, 0.75), xytext=(0.25, 0.65), arrowprops=arrow_params)
    ax.annotate("", xy=(0.25, 0.55), xytext=(0.25, 0.45), arrowprops=arrow_params)
    ax.annotate("", xy=(0.25, 0.35), xytext=(0.25, 0.25), arrowprops=arrow_params)
    ax.annotate("", xy=(0.4, 0.65), xytext=(0.5, 0.65), arrowprops=arrow_params)
    ax.annotate("", xy=(0.5, 0.55), xytext=(0.4, 0.55), arrowprops=arrow_params)
    ax.annotate("", xy=(0.65, 0.45), xytext=(0.8, 0.45), arrowprops=arrow_params)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title("整体流程图", fontproperties=font_prop)
    plt.show()

# 绘制双头网络结构图
def draw_dual_head_network():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制双头网络框架
    components = [
        ("教师网络\n（位置匹配分支）", (0.2, 0.7)),
        ("教师网络\n（类别匹配分支）", (0.2, 0.3)),
        ("学生网络\n（位置匹配分支）", (0.6, 0.7)),
        ("学生网络\n（类别匹配分支）", (0.6, 0.3)),
        ("三元组损失", (0.4, 0.7)),
        ("ArcFace损失\n与类别交叉熵损失", (0.4, 0.3)),
        ("蒸馏损失\n（logits和特征）", (0.8, 0.5))
    ]
    
    for component, (x, y) in components:
        ax.add_patch(patches.Rectangle((x, y), 0.15, 0.1, edgecolor='black', facecolor='lightgreen', lw=2))
        ax.text(x + 0.075, y + 0.05, component, ha='center', va='center', fontproperties=font_prop)
    
    # 绘制箭头
    arrow_params = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate("", xy=(0.35, 0.75), xytext=(0.4, 0.75), arrowprops=arrow_params)
    ax.annotate("", xy=(0.35, 0.35), xytext=(0.4, 0.35), arrowprops=arrow_params)
    ax.annotate("", xy=(0.55, 0.75), xytext=(0.6, 0.75), arrowprops=arrow_params)
    ax.annotate("", xy=(0.55, 0.35), xytext=(0.6, 0.35), arrowprops=arrow_params)
    ax.annotate("", xy=(0.55, 0.45), xytext=(0.8, 0.55), arrowprops=arrow_params)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title("双头网络结构图", fontproperties=font_prop)
    plt.show()

# 生成图表
draw_process_flow()
draw_dual_head_network()
