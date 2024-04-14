import scrapy
from bs4 import BeautifulSoup, NavigableString, Tag
import re

from urllib.parse import urlencode

def get_zenrows_api_url(url, api_key):
    payload = {
        'url': url,
        'js_render': 'true',
        'antibot': 'true',
        'premium_proxy': 'true'
    }
    # Construct the API URL by appending the encoded payload to the base URL with the API key
    api_url = f'https://api.zenrows.com/v1/?apikey={api_key}&{urlencode(payload)}'
    return api_url
class AdiletzanSpider(scrapy.Spider):
    name = "AdiletZan"
    # allowed_domains = [""]
    # start_urls = ["https://kodeksy-kz.com/ka/ugolovnyj_kodeks/1.htm"]
    start_urls = ["https://adilet.zan.kz/kaz/index/docs"]
    d = {}
    title = ""
    check = []
    names = []
    def parse(self, response):
        # self.logger.info(f'RESPONSE: {response.text}')
        hrefs = response.css('div.serp a').getall()
        links = []
        for html_snippet in hrefs:

            soup = BeautifulSoup(html_snippet, 'html.parser')#ll

            href_value = soup.find('a')['href']
            links.append(href_value)

        # links = ['/rus/docs/K030000481_']
        # links = ['/rus/docs/K1700000120']
        for link in links:
            yield response.follow(
                f'https://adilet.zan.kz{link}',
                callback=self.parse_article
            )
        # names = response.css("div.post_holder").getall()
        # for html_content in names:
        #     soup = BeautifulSoup(html_content, 'html.parser')
        #     extracted_text = soup.get_text().strip()
        #     cleaned_text = re.sub(r'\n+', ' ', extracted_text)
        #     cleaned_text = re.sub(r'^\d+\.', '', cleaned_text).strip()
        #     cleaned_name = re.sub(r'[\\/*?:"<>|\r\n]', " ", cleaned_text)
        #     self.check.append(''.join(cleaned_name.split()))
        next_page = response.css('a.nextpostslink::attr(href)').get()
        # next_page = None
        if next_page is not None:
            next_page_url = 'https://adilet.zan.kz' + next_page
            yield response.follow(next_page_url, callback=self.parse)
        else:
            with open('C:/Users/aliha/Desktop/NLP/parser/zakon/names/names6.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.names))




    def parse_article(self, response):
        html_content = response.css('[class="container_alpha slogan"]').get()
        soup = BeautifulSoup(html_content, 'html.parser')
        self.title = soup.get_text(separator='\n', strip=True)

        html_content = response.css("div.gs_12").get()
        if 'Күшін жойған' in self.title:
            return
        soup = BeautifulSoup(html_content, 'html.parser')

        articles = soup.find_all('article')
        contents = []
        for article in articles:  # Iterate over each article
            for element in article.children:  # Use .descendants to iterate through all children, recursively
                if element.name == 'table':
                    for row in element.find_all('tr'):
                        row_data = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                        contents.append('\t|\t'.join(row_data) + '\n')
                    contents.append('\n')

                elif element.name == 'a':
                    link_text = element.get_text(strip=True)
                    if link_text:
                        contents.append(link_text + ' ')
                elif element.name == 'span':
                    span_text = element.get_text(strip=True)
                    if span_text:
                        contents.append(span_text + ' ')
                    # print(span_text)
                #
                else:
                    text_content = element.get_text(strip=True, separator=' ')
                    contents.append(text_content + '\n')
        cleaned_name = re.sub(r'[\\/*?:"<>|\r\n]', " ", self.title)
        data = {
            cleaned_name: ''.join(contents)
        }
        yield data
        # self.d = data
        # self.to_txt()

    # def to_txt(self):

    #     temp = self.d['title'] + '\n\n' + self.d['text']
    #     dir = 'C:/Users/aliha/Desktop/NLP/parser/zakon/check/'
    #     cleaned_name = re.sub(r'[\\/*?:"<>|\r\n]', " ", self.title)[:100] + ".txt"
    #     self.names.append(cleaned_name)
    #     path = dir + cleaned_name
    #     with open(f'{path}', 'w', encoding='utf-8') as f:
    #         f.write(temp)


#