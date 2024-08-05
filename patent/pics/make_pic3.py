from graphviz import Digraph

dot = Digraph()

# 添加节点
dot.node('A', '输入图像', shape='box')
dot.node('B', '特征提取\n(骨干网络)', shape='box')
dot.node('C', '双头输出\n(特征匹配和分类)', shape='box')
dot.node('D', '特征匹配\n(余弦距离计算)', shape='box')
dot.node('E', '分类\n(类别概率分布)', shape='box')
dot.node('F', '匹配结果输出\n(相似度和类别)', shape='box')

# 添加边
dot.edges(['AB', 'BC', 'CD', 'CE', 'DF'])

# 配置图属性
dot.attr(size='10,8', rankdir='TB', splines='line')
dot.render('patent_pic_3_图像匹配和分类子模块流程图', format='png', cleanup=True)
dot.view()
