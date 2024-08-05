from graphviz import Digraph

dot = Digraph()

# 添加节点
dot.node('A', '输入图像', shape='box')
dot.node('B', '尺寸调整\n(保持比例)', shape='box')
dot.node('C', '关键点检测\n(提取特征)', shape='box')
dot.node('D', '几何变换\n(平移/旋转)', shape='box')
dot.node('E', '对齐图像\n(统一坐标)', shape='box')

# 添加边
dot.edges(['AB', 'BC', 'CD', 'DE'])

# 配置图属性
dot.attr(size='10,8', rankdir='TB', splines='line')
dot.render('patent_pic_6_图像尺寸对齐预处理示意图', format='png', cleanup=True)
dot.view()
