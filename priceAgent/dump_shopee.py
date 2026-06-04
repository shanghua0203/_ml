import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url='https://shopee.tw/%E7%8F%BE%E8%B2%A8%E2%9C%A8IVE-%E4%B8%89%E6%9C%9F-%E6%9C%83%E5%93%A1%E7%A6%AE-%E5%9B%BA%E9%85%8D-%E2%80%BC%EF%B8%8F%E7%84%A1%E5%B0%8F%E5%8D%A1%E5%92%8C%E6%9C%83%E5%93%A1%E5%8D%A1%E2%80%BC%EF%B8%8F%E5%AE%89%E5%85%AA%E7%9C%9F-%E9%87%91%E7%A7%8B%E5%A4%A9-%E7%9B%B4%E4%BA%95%E6%80%9C-%E5%BC%B5%E5%93%A1%E7%91%9B-%E9%87%91%E5%BF%97%E5%9E%A3-%E6%9D%8E%E8%B3%A2%E7%91%9E-i.970339123.24086596541')
        with open('shopee_dump.md', 'w', encoding='utf-8') as f:
            f.write(result.markdown)
        print('Dumped to shopee_dump.md')

asyncio.run(main())
