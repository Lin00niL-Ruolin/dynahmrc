#!/usr/bin/env python3
"""生成有效的 .docx 文档 - 含代码块、表格、图片"""
import zipfile, os, re, xml.etree.ElementTree as ET

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCX_FILE = os.path.join(OUT_DIR, 'DynaHMRC_毕业设计说明书.docx')

def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def make_p(text, style=None, bold=False, center=False, mono=False):
    xml = '<w:p><w:pPr>'
    if center: xml += '<w:jc w:val="center"/>'
    if style: xml += f'<w:pStyle w:val="{style}"/>'
    xml += '<w:spacing w:line="320" w:lineRule="auto"/></w:pPr>'
    xml += '<w:r><w:rPr>'
    if bold: xml += '<w:b/>'
    if mono: xml += '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="16"/>'
    else: xml += '<w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体"/><w:sz w:val="21"/>'
    xml += '</w:rPr>'
    xml += f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
    return xml

def make_table(headers, rows):
    xml = '<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/></w:tblPr>'
    xml += '<w:tr>'
    for h in headers:
        xml += '<w:tc><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>' + esc(h) + '</w:t></w:r></w:p></w:tc>'
    xml += '</w:tr>'
    for row in rows:
        xml += '<w:tr>'
        for c in row:
            xml += '<w:tc><w:p><w:r><w:t>' + esc(str(c)) + '</w:t></w:r></w:p></w:tc>'
        xml += '</w:tr>'
    xml += '</w:tbl>'
    return xml

def generate():
    with open(os.path.join(OUT_DIR, 'DynaHMRC_毕业设计说明书.md'), 'r', encoding='utf-8') as f:
        md = f.read()
    
    lines = md.replace('\r\n', '\n').split('\n')
    body = []
    in_code = False
    in_table = False
    code_lines = []
    table_data = []
    code_label = ''
    is_mermaid = False
    
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        
        # Handle code blocks
        if s.startswith('```'):
            if not in_code:
                in_code = True
                code_lines = []
                is_mermaid = 'mermaid' in s
                # Check if previous line has a code label like "代码清单X-X"
                if i > 1:
                    prev = lines[i-1].strip()
                    if '代码清单' in prev:
                        code_label = prev
                i += 1
                continue
            else:
                # End of code block
                if code_lines and not is_mermaid:
                    if code_label:
                        body.append(make_p(code_label.replace('**', ''), bold=True))
                        code_label = ''
                    body.append(make_p('[代码]'))
                    # Output code as monospace paragraphs
                    for cl in code_lines:
                        if cl.strip():
                            body.append(make_p(cl, mono=True))
                in_code = False
                code_lines = []
                i += 1
                continue
        
        if in_code:
            code_lines.append(s)
            i += 1
            continue
        
        # Page breaks
        if 'page-break' in s:
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
            i += 1
            continue
        
        # Tables
        if '|' in s and len(s) > 5:
            cols = [c.strip() for c in s.split('|') if c.strip()]
            if cols and not any(c.startswith(':') or c.startswith('-') for c in cols) and len(cols) >= 2:
                if not in_table: table_data = [cols]
                else: table_data.append(cols)
                in_table = True
                i += 1
                continue
        if in_table and table_data:
            if len(table_data) >= 2: body.append(make_table(table_data[0], table_data[1:]))
            table_data = []; in_table = False
        
        if not s: i += 1; continue
        
        # Headings
        if s.startswith('# '):
            body.append(make_p(s[2:], bold=True, center=True))
        elif s.startswith('## '):
            body.append(make_p(s[3:], style='a2'))
        elif s.startswith('### '):
            body.append(make_p(s[4:], style='a3'))
        elif s.startswith('**图') or s.startswith('**表'):
            body.append(make_p(s.strip('*'), center=True))
        elif s.startswith('> '):
            body.append(make_p(s[2:]))
        elif s.startswith('$$'):
            # Skip math formulas
            while i < len(lines) and not lines[i].strip().startswith('$$'):
                i += 1
            i += 1
            continue
        elif s.startswith('```mermaid'):
            # Skip entire mermaid block
            while i < len(lines) and not lines[i].strip().startswith('```'):
                i += 1
            i += 1
            continue
        else:
            text = s.replace('**', '').replace('*', '')
            if len(text) > 3:
                body.append(make_p(text))
        
        i += 1
    
    # Flush table
    if in_table and table_data and len(table_data) >= 2:
        body.append(make_table(table_data[0], table_data[1:]))
    
    body_xml = '\n'.join(body)
    
    doc_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {body_xml}
  </w:body>
</w:document>'''
    
    # Validate
    try:
        ET.fromstring(doc_xml.encode('utf-8'))
    except Exception as e:
        # Find issue
        for j, line in enumerate(doc_xml.split('\n')):
            try:
                ET.fromstring(('<root>' + line + '</root>').encode('utf-8'))
            except:
                print(f'  ⚠️ 行{j}: ...{line[-60:]}')
        raise e
    
    # Package
    ct = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="a2">
    <w:name w:val="heading 2"/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="28"/><w:rFonts w:eastAsia="黑体"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="a3">
    <w:name w:val="heading 3"/><w:pPr><w:spacing w:before="200" w:after="100"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="24"/><w:rFonts w:eastAsia="黑体"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr><w:spacing w:line="360"/></w:pPr>
    <w:rPr><w:sz w:val="21"/><w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体"/></w:rPr>
  </w:style>
</w:styles>'''
    
    with zipfile.ZipFile(DOCX_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', ct.encode('utf-8'))
        zf.writestr('_rels/.rels', rels.encode('utf-8'))
        zf.writestr('word/_rels/document.xml.rels', doc_rels.encode('utf-8'))
        zf.writestr('word/styles.xml', styles.encode('utf-8'))
        zf.writestr('word/document.xml', doc_xml.encode('utf-8'))
    
    # Count code blocks
    n_codes = body_xml.count('[代码]')
    print(f'✅ 已生成: {DOCX_FILE}')
    para_count = body_xml.count(chr(60) + chr(119) + chr(58) + chr(112))
    print(f"   段落数: {{para_count}}")
    print(f'   代码块数: {n_codes}')

if __name__ == '__main__':
    generate()
