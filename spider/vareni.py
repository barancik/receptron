import scrapy
from bs4 import BeautifulSoup

def clean_recipe(recipe):
    soup = BeautifulSoup(recipe,'lxml')
    paragraphs = [d.get_text().strip() for d in soup.find_all("p")[1:-1]]
    text = " ".join(paragraphs)
    #text = soup.get_text().replace('\n',' ').strip(' ').replace('  ',' ')
    return text.replace("\r\n"," ") #.replace(u'\xa0', u' ').replace(u'\xad', '')

class QuotesSpider(scrapy.Spider):
    name = "vareni"
    start_urls = [
        'http://recepty.vareni.cz/niva-dip/',
    ]

    def parse(self, response):
        # name = response.css('title::text').extract_first()
        recipe = response.xpath('//div[@id="recipe-procedure"]').extract_first()
        if recipe:
            yield {
                'name': response.css('title::text').extract_first(),
                'recept': clean_recipe(recipe),
            }
        next_pages = response.css('a::attr(href)').re(r'http://recepty.vareni.cz/.*')
        for next_page in next_pages:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)