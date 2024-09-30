from graphviz import Digraph

dot = Digraph()

# 添加节点
dot.node('A', '大尺寸图像输入', shape='box')
dot.node('B', '滑窗检测\n(重叠窗口)', shape='box')
dot.node('C', '局域坐标系\n(窗口内检测)', shape='box')
dot.node('D', '坐标转换\n(局域到全局)', shape='box')
dot.node('E', '全局坐标系\n(合并检测结果)', shape='box')
dot.node('F', '非极大值抑制\n(去除冗余)', shape='box')
dot.node('G', '结果输出\n(最终检测)', shape='box')

# 添加边
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

# 配置图属性
dot.attr(size='10,8', rankdir='TB', splines='line')
dot.render('patent_pic_5_滑窗检测和坐标转换流程图', format='png', cleanup=True)
dot.view()
