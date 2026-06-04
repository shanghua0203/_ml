#!/usr/bin/env python3
"""本地端 AI 比價購物助理 - 主程式"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Coroutine, Any

from langchain_ollama import ChatOllama

# 將專案根目錄加入 sys.path，確保直接執行腳本時能正確載入 src package
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.comparison_agent import ComparisonAgent
from src.agents.scrape_agent import ScrapeAgent
from src.core.cache import SearchCache
from src.core.config import settings
from src.utils.db import db


def print_header():
    """印出程式標頭"""
    print("=" * 60)
    print("本地端 AI 比價購物助理")
    print("=" * 60)
    print()


async def run_comparison(query: str, output_json: bool = False):
    """執行比價流程"""
    print_header()

    # 驗證設定
    errors = settings.validate()
    if errors:
        print("❌ 設定錯誤：")
        for error in errors:
            print(f"   - {error}")
        if not output_json:
            print("\n請設定環境變數後再試一次。")
        return 1

    print(f"🔍 搜尋商品：{query}")
    print("-" * 40)

    try:
        async with ComparisonAgent() as agent:
            result = await agent.compare(query)

        if output_json:
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        else:
            _print_result(result)

        return 0

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"❌ 發生錯誤：{e}")
        return 1


def _print_result(result):
    """印出比價結果"""
    print()
    print("=" * 60)
    print(f"比價結果：{result.product_name}")
    print("=" * 60)

    # 最推薦
    if result.most_recommended:
        print("\n🏆 最推薦")
        print(f"   名稱：{result.most_recommended.get('name', 'N/A')}")
        print(f"   價格：NT${result.most_recommended.get('price', 0):,.0f}")
        print(f"   平台：{result.most_recommended.get('source', 'N/A')}")
        print(f"   風險：{result.most_recommended.get('risk_level', 'N/A')}")
        print(f"   安全分數：{result.most_recommended.get('safety_score', 0):.0f}/100")

    # 最便宜
    if result.cheapest:
        print("\n💰 最便宜")
        print(f"   名稱：{result.cheapest.get('name', 'N/A')}")
        print(f"   價格：NT${result.cheapest.get('price', 0):,.0f}")
        print(f"   平台：{result.cheapest.get('source', 'N/A')}")

    # 最安全
    if result.safest:
        print("\n🛡️  最安全")
        print(f"   名稱：{result.safest.get('name', 'N/A')}")
        print(f"   價格：NT${result.safest.get('price', 0):,.0f}")
        print(f"   風險：{result.safest.get('risk_level', 'N/A')}")
        print(f"   安全分數：{result.safest.get('safety_score', 0):.0f}/100")

    print("\n" + "-" * 60)
    print("📝 完整比較列表：")
    print("-" * 60)

    for i, rec in enumerate(result.recommendations, 1):
        print(f"\n{i}. {rec.get('name', 'N/A')}")
        print(f"   價格：NT${rec.get('price', 0):,.0f}")
        print(f"   平台：{rec.get('source', 'N/A')}")
        print(f"   風險等級：{rec.get('risk_level', 'N/A')}")
        print(f"   安全分數：{rec.get('safety_score', 0):.0f}/100")
        print(f"   網址：{rec.get('url', 'N/A')}")
        if rec.get('scrape_failed'):
            print(f"   ⚠️  抓取失敗")

    # 輸出 LLM 建議
    if result.llm_recommendation:
        print("\n" + "=" * 60)
        print("🤖 Ollama AI 購買建議：")
        print("-" * 60)
        print(result.llm_recommendation)


def show_history(limit: int = 5):
    """顯示歷史紀錄"""
    print_header()
    print("🛒 歷史比價紀錄")
    print("=" * 60)

    async def _show_history():
        history = await db.get_history(limit=limit)
        if not history:
            print("\n尚無比價紀錄。")
            return

        for record in history:
            print(f"\n📅 {record['created_at']}")
            print(f"   搜尋：{record['search_query']}")
            print(f"   商品：{record['product_name']}")
            print(f"   價格：NT${record['price']:,.0f}")
            print(f"   平台：{record['source_platform']}")
            print(f"   風險：{record['risk_level']}")
            if record['is_recommendation']:
                print("   ⭐ 本場推薦")
    asyncio.run(_show_history())


def show_stats():
    """顯示統計資料"""
    print_header()
    print("📊 統計資料")
    print("=" * 60)

    async def _show_stats():
        stats = await db.get_stats()
        print(f"\n總比價紀錄：{stats['total_shopping_records']:,}")
        print(f"總搜尋次數：{stats['total_searches']:,}")
    asyncio.run(_show_stats())


def clear_cache():
    """清除快取"""
    cache = SearchCache()
    count = cache.clear_expired()
    print(f"✅ 已清除 {count} 筆過期快取")


async def run_single_scrape(url: str):
    """單一網頁抓取"""
    print_header()
    print(f"網頁抓取：{url}")
    print("-" * 40)

    async with ScrapeAgent() as agent:
        result = await agent.scrape(url)

    print(f"\n標題：{result.title}")
    print(f"抓取時間：{result.scraped_at}")
    print(f"\n內容（前 500 字）：")
    print(result.markdown[:500])
    if len(result.markdown) > 500:
        print("...")


def safe_run(coro: Coroutine) -> Any:
    """執行 async 協程"""
    # 由於 ComparisonAgent 已在 finally 正確關閉 Playwright (ScrapeAgent.__aexit__)
    # 此處不再需要手動管理 pending tasks，直接使用 asyncio.run 即可正常退出
    return asyncio.run(coro)


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="本地端 AI 比價購物助理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python src/main.py --search "iphone 15"
  python src/main.py --search "iphone 15" --json
  python src/main.py --history
  python src/main.py --stats
  python src/main.py --scrape https://example.com
  python src/main.py --clear-cache
        """
    )

    parser.add_argument("--search", "-s", metavar="QUERY", help="搜尋商品")
    parser.add_argument("--json", "-j", action="store_true", help="以 JSON 格式輸出")
    parser.add_argument("--history", action="store_true", help="顯示歷史紀錄")
    parser.add_argument("--stats", action="store_true", help="顯示統計資料")
    parser.add_argument("--scrape", metavar="URL", help="單一網頁抓取")
    parser.add_argument("--clear-cache", action="store_true", help="清除過期快取")
    parser.add_argument("--env", action="store_true", help="顯示目前環境設定")

    args = parser.parse_args()

    if args.env:
        print("環境設定：")
        print(f"  Ollama URL: {settings.OLLAMA_BASE_URL}")
        print(f"  Ollama Model: {settings.OLLAMA_MODEL}")
        print(f"  Brave API Key: {'*' * 10 + settings.BRAVE_API_KEY[-4:] if settings.BRAVE_API_KEY else '未設定'}")
        print(f"  Database: {settings.DATABASE_PATH}")
        return 0

    if args.clear_cache:
        clear_cache()
        return 0

    if args.history:
        show_history()
        return 0

    if args.stats:
        show_stats()
        return 0

    if args.scrape:
        return safe_run(run_single_scrape(args.scrape))

    if args.search:
        return safe_run(run_comparison(args.search, args.json))

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
