"""Draw the implemented C3 data flow as editable SVG (stdlib only).

Optional exports: run with a Python environment containing cairosvg.
"""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
W, H = 2040, 1440
C = {'ink':'#172638', 'muted':'#526275', 'line':'#8896a6',
     'blue':'#2463a4', 'purple':'#8054aa', 'teal':'#167d78',
     'gray':'#6f7c8a'}
parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">原报告与功能假设双路语义检索及证据重排流程</title>
<desc id="desc">当前 C3 方法。两路共享代码向量库，假设前40个文件与原文补足至50个文件，每文件原文第一块与假设补足至三块。原报告重排，文件Max聚合。标签仅用于评价。微调为计划扩展。</desc>
<defs>''']
for key in ['blue','purple','teal','gray']:
    parts.append(f'<marker id="arrow-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="{C[key]}"/></marker>')
parts.append('</defs><rect width="2040" height="1440" fill="white"/>')

def text(x, y, value, size=24, color='ink', weight=400, anchor='start'):
    fill = C.get(color,color)
    parts.append(f'<text x="{x}" y="{y}" font-family="Noto Sans CJK SC, Microsoft YaHei, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(value)}</text>')

def rect(x,y,w,h,fill='#fff',stroke='#d3dce5',dash=False,r=14):
    d=' stroke-dasharray="9 7"' if dash else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>')

def node(x,y,w,h,title,lines,color='gray',fill='#fff'):
    rect(x,y,w,h,fill,C[color])
    text(x+22,y+40,title,27,color,700)
    for i,line in enumerate(lines):
        text(x+22,y+77+i*30,line,22,'muted')

def edge(points,color='gray',dashed=False,width=3):
    d='M '+' L '.join(f'{x} {y}' for x,y in points)
    # White underlay makes crossings visibly separate, without junction dots.
    parts.append(f'<path d="{d}" fill="none" stroke="white" stroke-width="{width+5}" stroke-linejoin="round"/>')
    dash=' stroke-dasharray="8 7"' if dashed else ''
    parts.append(f'<path d="{d}" fill="none" stroke="{C[color]}" stroke-width="{width}" stroke-linejoin="round" marker-end="url(#arrow-{color})"{dash}/>')

text(60,66,'原报告与功能假设的双路语义检索及证据重排',40,'ink',700)
text(60,108,'文件级缺陷定位 · 当前 C3 方法的数据流',24,'muted')
for x,label in [(60,'01  查询构建'),(470,'02  共享索引检索'),(960,'03  候选与证据组合'),(1510,'04  重排与评价')]:
    text(x,180,label,27,'ink',700)
    parts.append(f'<path d="M{x} 201 H{x+min(440,1980-x)}" stroke="#dce3ea" stroke-width="2"/>')

# Offline source preparation panel and shared-index connectors.
rect(60,1040,1340,190,'#f6f8fa','#dbe3eb')
text(82,1078,'共享代码准备  ·  两路复用同一代码块和向量库',24,'muted',500)
edge([(1180,1115),(1180,1000),(435,1000),(435,430),(470,430)],'gray',True,2.5)
edge([(435,1000),(435,910),(470,910)],'gray',True,2.5)

# Online dataflow.
edge([(360,370),(470,370)],'blue')
edge([(210,430),(210,590)],'purple')
text(228,520,'仅报告输入',21,'purple')
edge([(210,710),(210,790)],'purple')
edge([(360,855),(470,855)],'purple')
edge([(830,370),(900,370),(900,490),(960,490)],'blue')
edge([(830,855),(900,855),(900,550),(960,550)],'purple')
edge([(1180,595),(1180,700)],'teal')
text(1200,637,'候选文件',21,'teal')
text(1200,667,'＋两路块级分数',21,'teal')
edge([(1400,790),(1450,790),(1450,555),(1510,555)],'teal')
edge([(1745,630),(1745,700)],'teal')
text(1765,660,'每块相关性得分',21,'teal')
edge([(1745,800),(1745,865)],'teal')
edge([(1745,965),(1745,1070)],'teal')
text(1765,1020,'前 k 个文件',21,'teal')
edge([(1400,1310),(1450,1310),(1450,1135),(1510,1135)],'gray',True)

# Original-report bypass to the reranker, independent of hypotheses.
edge([(210,310),(210,250),(1745,250),(1745,440)],'blue')
text(1040,235,'重排 Query：原报告（沿用既有截断）',23,'blue',500,'middle')

node(60,310,300,120,'原始缺陷报告 r',['保留报告中的直接线索'],'blue','#f1f7fd')
node(60,590,300,120,'功能假设生成',['Qwen3-8B','不读取源码或修复标签'],'purple','#f8f4fc')
node(60,790,300,130,'功能假设 h',['context · functionality','logic'],'purple','#f8f4fc')
node(470,310,360,155,'原报告语义检索',['Qwen3-Embedding-0.6B','块级相似度 → 文件 Max','输出文件排名与块级分数'],'blue','#f1f7fd')
node(470,790,360,155,'功能假设语义检索',['同一编码器与代码向量库','块级相似度 → 文件 Max','输出文件排名与块级分数'],'purple','#f8f4fc')
node(960,440,440,155,'固定预算候选合并',['功能假设 Top-40 优先','原报告排名去重补足至 50','保留两路块级分数供后续选块'],'teal','#f0f9f7')
node(960,700,440,180,'C3 双视角代码证据选择',['对每个候选文件：','① 保留原报告 Top-1 代码块','② 按功能假设排名去重补足','每文件最多 3 个不同代码块'],'teal','#f0f9f7')
node(1510,440,470,190,'原报告与代码块相关性重排',['Qwen3-Reranker-0.6B','Query = 原报告；Document = 代码块','块得分 = logit(yes) − logit(no)','功能假设文本不直接进入重排器'],'teal','#f0f9f7')
node(1510,700,470,100,'文件级得分聚合',['文件得分 = 所选代码块得分的 Max'],'teal')
node(1510,865,470,100,'输出候选文件排名',['得分降序；同分按原报告语义排名'],'teal')
node(1510,1070,470,130,'文件级定位评价',['MAP@10 · MRR@10 · Top-1/5/10','MeanRecall@10；失败按零计入'],'gray','#f6f8fa')
node(1130,1270,270,80,'修复文件标签',[],'gray','#f6f8fa')
text(1265,1384,'仅用于评价',22,'gray',500,'middle')

# Code preparation, with content vectors shared by both query routes.
node(85,1105,280,100,'修复前源码快照',['候选源码文件'],'gray')
node(440,1105,330,100,'AST 分块与结构前缀',['AST64 · 1500 字符初始预算'],'gray')
node(845,1105,525,100,'代码编码与共享向量库',['Qwen3-Embedding-0.6B · 代码向量可复用'],'gray')
edge([(365,1155),(440,1155)],'gray')
edge([(770,1155),(845,1155)],'gray')

rect(1510,1260,470,100,'#fafafa','#a2aab4',True)
text(1532,1298,'计划扩展：重排器任务微调',25,'muted',600)
text(1532,1333,'仅更新模型权重，不属于当前推理步骤',21,'muted')
text(60,1295,'实线：处理流程    灰色虚线：共享代码索引或评价标签',23,'muted')
text(60,1335,'候选或代码块不足时保留全部；生成与检索失败按既定协议记录。',22,'muted')
text(60,1395,'C3  /  原报告 + 功能假设双路语义检索',20,'muted')

parts.append('</svg>')
svg='\n'.join(parts)
path=OUT/'c3-pipeline.svg'
path.write_text(svg,encoding='utf-8')
print(path)
try:
    import cairosvg
except ImportError:
    print('SVG saved. Install cairosvg in an isolated environment for PDF/PNG export.')
else:
    cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(OUT/'c3-pipeline.pdf'))
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/'c3-pipeline.png'))
    print('PDF and PNG exported.')
