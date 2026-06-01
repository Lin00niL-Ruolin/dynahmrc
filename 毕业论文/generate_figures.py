#!/usr/bin/env python3
"""用 Pillow 生成论文用简图"""
from PIL import Image, ImageDraw, ImageFont
import os

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Colors
C_BG = (255, 255, 255)
C_FG = (0, 0, 0)
C_BLUE = (70, 130, 180)
C_LBLUE = (200, 220, 240)
C_GREEN = (100, 160, 100)
C_ORANGE = (220, 160, 80)
C_GRAY = (200, 200, 200)
C_DGRAY = (100, 100, 100)

def font(size=14):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', size)
    except:
        try:
            return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size)
        except:
            return ImageFont.load_default()

def draw_box(d, xy, text, fill=C_LBLUE, border=C_BLUE):
    d.rounded_rectangle(xy, radius=5, fill=fill, outline=border, width=2)
    cx = (xy[0] + xy[2]) // 2
    cy = (xy[1] + xy[3]) // 2
    f = font(13)
    bbox = d.textbbox((0,0), text, font=f)
    d.text((cx - (bbox[2]-bbox[0])//2, cy - (bbox[3]-bbox[1])//2), text, fill=C_FG, font=f)

def draw_arrow(d, x1, y1, x2, y2):
    d.line([(x1, y1), (x2, y2)], fill=C_DGRAY, width=2)
    # Arrowhead
    if y2 > y1:  # down
        d.polygon([(x2-5, y2-8), (x2+5, y2-8), (x2, y2)], fill=C_DGRAY)

def fig2_1_api_flow():
    """图2-1：API调用流程图"""
    img = Image.new('RGB', (500, 250), C_BG)
    d = ImageDraw.Draw(img)
    
    draw_box(d, (50, 20, 200, 70), '应用系统')
    draw_box(d, (50, 100, 200, 150), 'DeepSeek SDK')
    draw_box(d, (250, 100, 400, 150), 'DeepSeek API')
    draw_box(d, (300, 20, 450, 70), '大语言模型')
    
    draw_arrow(d, 125, 70, 125, 100)
    d.text((130, 80), '构造messages', fill=C_DGRAY, font=font(10))
    
    draw_arrow(d, 200, 125, 250, 125)
    d.text((210, 115), 'POST /chat', fill=C_DGRAY, font=font(10))
    
    draw_arrow(d, 325, 100, 325, 70)
    d.text((330, 80), '推理回复', fill=C_DGRAY, font=font(10))
    
    draw_arrow(d, 200, 130, 250, 130)
    d.text((210, 130), '返回JSON', fill=C_DGRAY, font=font(10))
    
    img.save(os.path.join(FIG_DIR, 'fig2-1.png'))
    print('✅ fig2-1.png')

def fig4_1_architecture():
    """图4-1：系统架构图"""
    img = Image.new('RGB', (500, 420), C_BG)
    d = ImageDraw.Draw(img)
    
    # Layers background
    labels = ['前端展示层 (React + Vite)', '后端服务层 (Node.js + Express)', 'LLM推理层 (DeepSeek API)', '3D仿真层 (PyBullet)']
    colors = [(220,235,255), (255,245,220), (220,255,220), (255,225,225)]
    
    y = 10
    boxes = [
        ('配置面板', '仿真画布', '对话日志'),
        ('DynaHMRC引擎', 'Agent管理器', '仿真环境', 'WebSocket'),
        ('DeepSeek API',),
        ('BestMan / PyBullet',),
    ]
    
    for li, (label, clr) in enumerate(zip(labels, colors)):
        h = 80 if li < 2 else 50
        d.rectangle([(10, y), (490, y+h)], fill=clr, outline=C_BLUE, width=1)
        d.text((15, y+2), label, fill=C_BLUE, font=font(10))
        
        # Sub boxes
        bx = 20
        for btext in boxes[li]:
            bw = 130
            draw_box(d, (bx, y+18, bx+bw, y+h-5), btext, fill=(255,255,255), border=C_GRAY)
            bx += bw + 10
        
        y += h + 5
    
    # Connection arrows between layers
    for yy in [90, 170, 260]:
        draw_arrow(d, 250, yy, 250, yy+5)
    
    img.save(os.path.join(FIG_DIR, 'fig4-1.png'))
    print('✅ fig4-1.png')

def fig4_2_collaboration():
    """图4-2：四阶段协作流程图"""
    img = Image.new('RGB', (500, 350), C_BG)
    d = ImageDraw.Draw(img)
    
    stages = [
        ('阶段一：自我描述', '自我介绍', C_LBLUE),
        ('阶段二：任务分配与竞选', '分工方案+演讲', (255,230,200)),
        ('阶段三：领导选举', '投票选举领导者', (200,255,200)),
        ('阶段四：执行与反思', '动作→反馈→反思', (255,200,200)),
    ]
    
    y = 10
    for title, desc, clr in stages:
        d.rectangle([(30, y), (470, y+70)], fill=clr, outline=C_DGRAY, width=2)
        d.text((40, y+8), title, fill=C_FG, font=font(14))
        d.text((40, y+35), desc, fill=C_DGRAY, font=font(12))
        
        if y > 10:
            draw_arrow(d, 250, y-5, 250, y)
        
        y += 80
    
    img.save(os.path.join(FIG_DIR, 'fig4-2.png'))
    print('✅ fig4-2.png')

def fig4_3_entity():
    """图4-3：数据实体关系图"""
    img = Image.new('RGB', (500, 200), C_BG)
    d = ImageDraw.Draw(img)
    
    draw_box(d, (30, 60, 150, 120), 'DynaHMRC\n引擎', fill=(200,220,255))
    draw_box(d, (200, 20, 330, 80), 'Agent\n(机器人)', fill=(255,220,200))
    draw_box(d, (200, 110, 330, 170), '仿真环境\nSimEnvironment', fill=(200,255,200))
    draw_box(d, (380, 20, 480, 80), 'LLM\nAPI', fill=(220,200,255))
    
    d.line([(150, 80), (200, 50)], fill=C_DGRAY, width=2)
    d.text((160, 55), '管理', fill=C_DGRAY, font=font(10))
    d.line([(150, 100), (200, 140)], fill=C_DGRAY, width=2)
    d.text((160, 115), '维护', fill=C_DGRAY, font=font(10))
    d.line([(330, 50), (380, 50)], fill=C_DGRAY, width=2)
    d.text((345, 38), '调用', fill=C_DGRAY, font=font(10))
    
    img.save(os.path.join(FIG_DIR, 'fig4-3.png'))
    print('✅ fig4-3.png')

if __name__ == '__main__':
    print('生成论文插图...')
    fig2_1_api_flow()
    fig4_1_architecture()
    fig4_2_collaboration()
    fig4_3_entity()
    print('完成!')
