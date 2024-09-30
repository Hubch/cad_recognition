from graphviz import Digraph

dot = Digraph()

# 添加节点
dot.node('A', '特征提取\n(高维特征向量)', shape='box')
dot.node('B', '余弦距离计算\n(相似性度量)', shape='box')
dot.node('C', '相似度矩阵\n(特征对比较)', shape='box')
dot.node('D', '阈值优化\n(精度优化)', shape='box')
dot.node('E', '匹配结果输出\n(最终结果)', shape='box')

# 添加边
dot.edges(['AB', 'BC', 'CD', 'DE'])

# 配置图属性
dot.attr(size='10,8', rankdir='TB', splines='line')
dot.render('patent_pic_7_特征匹配子模块流程图', format='png', cleanup=True)
dot.view()
