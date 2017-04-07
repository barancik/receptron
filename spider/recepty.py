import scrapy
from bs4 import BeautifulSoup

def clean_recipe(recipe):
    soup = BeautifulSoup(recipe,'lxml')
    return soup.get_text().strip().replace("\n"," ")

class QuotesSpider(scrapy.Spider):
    name = "vareni"
    start_urls = [
        'http://www.recepty.cz/recept/domaci-sunkofleky-s-uzenym-151524',
    ]

    def parse(self, response):
        # name = response.css('title::text').extract_first()
        recept = response.xpath('//div[@class="wikiPreview"]').extract_first()
        if recept:
            yield {
                'name': response.css('title::text').extract_first(),
                'recept': clean_recipe(recept),
            }
        next_pages = set(response.css('a::attr(href)').re(r'/recept/.*[0-9][0-9]'))
        for next_page in next_pages:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)