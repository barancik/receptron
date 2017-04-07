import scrapy
from bs4 import BeautifulSoup

def clean_recipe(recipe):
    soup = BeautifulSoup(recipe,'lxml')
    text = soup.get_text().strip().replace('\n',' ').strip(' ').replace('  ',' ')
    return text

class QuotesSpider(scrapy.Spider):
    name = "labuznik"
    start_urls = [
        'https://www.toprecepty.cz/recept/51578-tuse-testoviny-s-parmazanem/',
        #  'http://www.labuznik.cz/recept/kakanek-neboli-kakan/',
    ]

    def parse(self, response):
        # name = response.css('title::text').extract_first()
        recipe = response.xpath('//ol[@itemprop="recipeInstructions"]').extract_first()
        if recipe:
            yield {
                'name': response.css('title::text').extract_first(),
                'recept': clean_recipe(recipe),
            }

        next_pages = response.css('a::attr(href)').re(r'https://www.toprecepty.cz/recept/.*')
        for next_page in next_pages:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)