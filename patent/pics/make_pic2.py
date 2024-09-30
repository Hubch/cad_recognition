from graphviz import Digraph

dot = Digraph()

# 添加节点
dot.node('A', '输入图像', shape='box')
dot.node('B', '数据预处理\n(增强和归一化)', shape='box')
dot.node('C', 'YOLOv5 模型', shape='box')
dot.node('D', '目标检测\n(检测关键目标)', shape='box')
dot.node('E', '非极大值抑制\n(去除冗余)', shape='box')
dot.node('F', '结果输出\n(目标位置和类别)', shape='box')

# 添加边
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

# 配置图属性
dot.attr(size='10,8', rankdir='TB', splines='line')
dot.render('patent_pic_2_目标检测子模块流程图', format='png', cleanup=True)
dot.view()
