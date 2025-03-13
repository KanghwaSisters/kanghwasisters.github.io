import json

basic_html = """---
layout: default
title: 구성원
weight: 2
permalink: /member/
---

<br>
# Members  
<br>  

### 강화시스터즈 동아리원을 소개합니다! 
역대 운영진과 동아리원들을 소개합니다!  
현재 활동 중인 동아리원은 흰색, 활동을 완료한 동아리원은 <b style="background-color:#f1f3f5;">회색</b>으로 구분되어 있습니다. 

<br>

### 어떤 정보를 확인할 수 있나요? 
✔️ 동아리원들의 <b style="background-color:#f6e705;">직책 + 전공</b>을 알 수 있어요.  
✔️ 흥미로운 동아리원이 있다면 <b style="background-color:#f6e705;">메일</b>로 연락할 수 있어요.  
✔️ <b style="background-color:#f6e705;">깃허브</b>를 통해 현재 활동을 볼 수 있어요.  
✔️ <b style="background-color:#f6e705;">태그</b>를 통해 동아리원들이 동아리 활동 중 작성한 글들을 확인할 수 있어요.  

<br>
"""


def load_members(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def generate_member_html(member):
    bgcolor = "#f1f3f5" if member.get("status", "active") == "inactive" else "white"
    selection = f"""{member['position']} <br>
{member['major']} <br><br>""" if member['position'] != '' else f"{member['major']} <br><br>"
    
    return f'''<td style="background-color:{bgcolor};">
<img src="../assets/image/members/{member['image']}" width=150/>
</td>
<td style="background-color:{bgcolor};">
<b style="font-size:15px">{member['name']} ({member['english_name']})</b> <hr>
{selection}
<a href="https://github.com/{member['github']}">
<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/>
</a>
<a href="mailto:{member['email']}">
<img src="https://img.shields.io/badge/gmail-EA4335?style=flat-square&logo=gmail&logoColor=white"/>
</a>
<a href="/blog/author#{member['name']}"> 
<img src="../assets/image/etc/tag.png" width=18/>
</a>
</td>
'''

def generate_member_table(batch_group, year, batch):
    html = ""
    html += f"""
<h4>📕 {batch_group}기 ({year})</h4>
<table style="width: 80%; border-collapse: collapse; table-layout: fixed;">
<tr>"""
    for i, member in enumerate(batch):
        html += generate_member_html(member)
        if (i + 1) % 2 == 0:
            html += "</tr><tr>"
    html += "</tr></table>  "
    
    return html

def generate_manager_table(batch):
    html = ""
    html += f"""<h4>👩‍💻 운영진</h4>
<table style="width: 80%; border-collapse: collapse; table-layout: fixed;">
<tr>"""
    for i, member in enumerate(batch):
        html += generate_member_html(member)
        if (i + 1) % 2 == 0:
            html += "</tr><tr>"
    html += "</tr></table>  "

    return html


if __name__ == "__main__":
    # load members
    management = load_members("./assets/member_manage/manager.json")
    batch1 = load_members("./assets/member_manage/batch1.json")
    batch2 = load_members("./assets/member_manage/batch2.json")
    batch3 = load_members("./assets/member_manage/batch3.json")

    # html
    html = ""
    html += basic_html
    html += generate_manager_table(management)
    html += "<br><br><br>"
    html += generate_member_table(1, '24-1', batch1)
    html += "<br><br><br>"
    html += generate_member_table(2, '24-2', batch2)
    html += "<br><br><br>"
    html += generate_member_table(3, '25-1', batch3)

    with open("/Users/ijimin/kanghwasisters.github.io/pages/members.md", "w", encoding="utf-8") as file:
        file.write(html)
    print("HTML 파일이 생성되었습니다.")
