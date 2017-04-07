import scrapy
from bs4 import BeautifulSoup

def clean_recipe(recipe):
    soup = BeautifulSoup(recipe,'lxml')
    text = soup.get_text().replace('\n',' ').strip(' ').replace('  ',' ')
    return text.replace(u'\xa0', u' ').replace(u'\xad', '')

class QuotesSpider(scrapy.Spider):
    name = "labuznik"
    start_urls = [
        'http://www.labuznik.cz/recept/pissaladiere/',
        #  'http://www.labuznik.cz/recept/kakanek-neboli-kakan/',
    ]

    def parse(self, response):
        # name = response.css('title::text').extract_first()
        recept = response.xpath('//div[@itemprop="recipeInstructions"]').extract_first()
        if recept:
            yield {
                'name': response.css('title::text').extract_first(),
                'recept': clean_recipe(recept),
            }

        next_pages = response.css('a::attr(href)').re(r'http://www.labuznik.cz/recept/.*')
        for next_page in next_pages:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)