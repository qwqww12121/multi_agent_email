import re
from html.parser import HTMLParser

class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[];self.links=[];self.hidden=0;self.anchor=None

    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','head'): self.hidden+=1
        if self.hidden:return
        attrs=dict(attrs)
        if tag in ('p','div','br','li','tr','blockquote'):self.parts.append('\n')
        if tag=='a':
            self.anchor=[attrs.get('href',''),''];self.links.append(self.anchor)

    def handle_endtag(self,tag):
        if tag in ('script','style','head'):self.hidden=max(0,self.hidden-1)
        if not self.hidden and tag in ('p','div','li','tr','blockquote'):self.parts.append('\n')
        if tag=='a':self.anchor=None

    def handle_data(self,data):
        if not self.hidden:
            self.parts.append(data)
            if self.anchor is not None:self.anchor[1]+=data

def normalize_space(text):
    text=text.replace('\r\n','\n').replace('\r','\n')
    text='\n'.join(re.sub(r'[^\S\n]+',' ',line).strip() for line in text.split('\n'))
    return re.sub(r'\n{3,}','\n\n',text).strip()

def visible_html(html):
    p=VisibleHTML();p.feed(html)
    return normalize_space(''.join(p.parts)),tuple((url,normalize_space(label)) for url,label in p.links)
