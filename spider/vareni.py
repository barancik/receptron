import scrapy
import re
from bs4 import BeautifulSoup
from scrapy import optional_features

optional_features.remove('boto')

def clean_recipe(recipe):
    soup = BeautifulSoup(recipe,'lxml')
    return soup.get_text()

    text = soup.get_text().replace('\n',' ').strip(' ').replace('  ',' ')
    return text.replace(u'\xa0', u' ').replace(u'\xad', '')

#def clean_recipe(recipe):
#    soup = BeautifulSoup(recipe,'lxml')
#    paragraphs = [d.get_text().strip() for d in soup.find_all("p")[1:-1]]
#    text = " ".join(paragraphs)
    #text = soup.get_text().replace('\n',' ').strip(' ').replace('  ',' ')
#    return text.replace("\r\n"," ") #.replace(u'\xa0', u' ').replace(u'\xad', '')

def get_ingredience(ingredients):
     soup = BeautifulSoup(ingredients,'lxml')
     ing = soup.find_all("dl")
     final = []
     for i in ing:
         halfway = i.getText().replace("\n"," ").strip()
         final.append(re.sub(" +"," ",halfway))
     return "\n".join(final)

class QuotesSpider(scrapy.Spider):
    name = "vareni"

    start_urls = [
        'http://recepty.vareni.cz/niva-dip/',
    ]

    def parse(self, response):
        # name = response.css('title::text').extract_first()
        recipe = response.xpath('//div[@id="recipe-procedure"]').extract_first()
        keywords = response.css('head').select('//meta[@name="keywords"]/@content').extract_first()
        ingredients = response.xpath('//div[@id="recipe-ingredients"]').extract_first()
        rating = clean_recipe(response.xpath('//div[@itemprop="rating"]').extract_first())
        typ = clean_recipe(response.xpath('//b[@itemprop="recipeType"]').extract_first())

        if recipe:
            yield {
                'name': response.css('title::text').extract_first(),
                'ingredients': get_ingredience(ingredients),
                'recept': clean_recipe(recipe),
                'rating':rating,
                'type':typ,
            }
        next_pages = set(response.css('a::attr(href)').re(r'/[a-z-]+/'))
        for next_page in next_pages:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
