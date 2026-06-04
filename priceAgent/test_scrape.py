import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url="https://24h.pchome.com.tw/prod/DCAIAL-A900G5K61")
        print("--- MARKDOWN LENGTH ---")
        print(len(result.markdown))
        
        lines = result.markdown.splitlines()
        print("--- LINES WITH DIGITS ---")
        for i, line in enumerate(lines):
            line = line.strip()
            if any(char.isdigit() for char in line) and ('$' in line or 'NT' in line or '元' in line or '9980' in line):
                print(f"{i}: {line}")

asyncio.run(main())
