import sys
import requests
import lxml.html

URL = "https://en.wikipedia.org/wiki/Wikipedia:Picture_of_the_day"

README = """
# 欢迎来到CB的Github 👋

<div align="center">
  <img height="137px" src="https://github-readme-stats.vercel.app/api?username=SuperCB&show_icons=true&theme=radical" />
   <img align="center" src="https://github-readme-stats.vercel.app/api/top-langs/?username=SuperCB&hide=javascript,html,cmake,tex&layout=compact&theme=swift" />
 
</div>


<div align="center">
    <img src="./contribution-snake/github-contribution-grid-snake.svg" />
</div>



## Picture of the day
<div align="center">
  <img width=400px src="{wiki_link}" />
</div>

>{wiki_content}

"""

print("download image...")
content = requests.get(URL).content
html = lxml.html.fromstring(content)
presentation_table = html.xpath("//table[@role='presentation']")[0]
a_tag = presentation_table.xpath(".//a[@class='image']")[0]
relative_link = a_tag.get("href")
title = a_tag.get("title")
image_src = a_tag.xpath("./img/@srcset")[0]
best_image = image_src.split(" ")[0]
print(f"{relative_link} {title}, image srcset:{image_src}")
print(f"best image: {best_image}")

content = requests.get(URL).content
html = lxml.html.fromstring(content)
presentation_table = html.xpath("//table[@role='presentation']")[0]
content = presentation_table.xpath('.//tbody/tr/td')[1]
ptext = content.xpath('.//p')[0]

contenttext = ptext.xpath('.//text()')

new_readme = README.format(
    wiki_link="https://" + best_image[2:],
    wiki_content=' '.join(contenttext)
)

with open("README.md", "w") as f:
    f.write(new_readme)


