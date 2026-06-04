"""比價比較 Agent - 整合搜尋、抓取、分析的完整流程"""
import asyncio
import json
import re
import time
from typing import Any, Optional
from pydantic import BaseModel, Field

import httpx
from langchain_ollama import ChatOllama

from ..agents.price_guard import PriceGuardAgent
from ..agents.scrape_agent import ScrapeAgent
from ..core.cache import SearchCache
from ..core.config import settings
from ..models.products import (
    ComparisonResult,
    FraudDetectionResult,
    PriceExtraction,
    RankedURLs,
)
from ..utils.db import db
from ..utils.scoring import ScoringEngine

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# 買賣平台網域白名單（優先抓取）
SHOPPING_DOMAINS = {
    "momo": ["momo", "momoshop"],
    "shopee": ["shopee"],
    "pchome": ["pchome", "techbang"],
    "yahoo": ["yahoo"],
    "rakuten": ["rakuten"],
    "books.com": ["books.com.tw"],
    "tmall": ["tmall", "taobao"],
    "amazon": ["amazon"],
}

# 避免抓取的非購物網站關鍵字
NON_SHOPPING_KEYWORDS = [
    "開箱", "測評", "評測", "推薦", "排行榜", "心得", "教學",
    "影片", "youtube", "vlog", " unboxing", "review",
    "news", "新聞", "公告", "說明", "faq", "幫助", "客服",
    " Blog ", " Blogspot ", " Medium ", " Wordpress ",
    "ptt", "批踢踢", "討論區", "討論板",
    "support", "支援", "技術規格", "規格", "手冊", "download",
    "apple", "wiki", "wikidata",
]

# 明確非購物網站的網域後綴
NON_SHOPPING_DOMAINS = [
    "apple.com", "apple.com.tw",
    "support.apple.com",
    "docs.google.com",
    "drive.google.com",
    "medium.com",
    "blogspot.com",
    "wordpress.com",
    "ptt.cc",
    "hackmd.io",
    "github.com",
    "wikipedia.org",
]


class ComparisonAgent:
    """比價比較 Agent - 整合所有流程"""

    def __init__(
        self,
        llm_model: str = None,
        temperature: float = 0.1,
        llm_instance: Any = None
    ):
        # 如果提供 LLM 實例，使用它；否則創建新的 ChatOllama
        if llm_instance is not None:
            self.llm = llm_instance
        else:
            self.llm = ChatOllama(
                model=llm_model or settings.OLLAMA_MODEL,
                temperature=temperature,
                base_url=settings.OLLAMA_BASE_URL
            )
        self.price_guard = PriceGuardAgent(self.llm)
        self.scrape_agent = ScrapeAgent()
        self.cache = SearchCache()

    async def __aenter__(self):
        """Async Context Manager 入口"""
        await self.scrape_agent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager 出口"""
        if self.scrape_agent:
            await self.scrape_agent.__aexit__(exc_type, exc_val, exc_tb)

    def _is_shopping_site(self, url: str, title: str = "") -> bool:
        """判斷是否為購物網站"""
        url_lower = url.lower()
        title_lower = title.lower() if title else ""

        # 1. 檢查非購物網站網域（高優先級）
        for domain in NON_SHOPPING_DOMAINS:
            if domain in url_lower:
                return False

        # 2. 檢查是否為白名單购物網站
        for domain_key, domain_patterns in SHOPPING_DOMAINS.items():
            if any(pattern in url_lower for pattern in domain_patterns):
                return True

        # 3. 檢查是否為非購物網站關鍵字
        for keyword in NON_SHOPPING_KEYWORDS:
            if keyword in title_lower or keyword in url_lower:
                return False

        # 4. 兜底：如果有明顯購物詞彙則視為購物網站
        shopping_words = ["產品", "商品", "購買", "特價", "優惠", "結帳", " cart ", "buy"]
        if any(word in url_lower or word in title_lower for word in shopping_words):
            return True

        # 5. 排除常見非購物網域
        excluded_tlds = [".php", ".html", ".asp", ".aspx", ".do", ".action"]
        if any(url_lower.endswith(tld) for tld in excluded_tlds):
            # 特殊處理：有些購物網站也用 .php，需要進一步檢查
            if not any(domain in url_lower for domain in ["momo", "shopee", "pchome", "yahoo"]):
                return False

        return False

    def _filter_search_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """過濾搜尋結果，優先選取購物網站"""
        shopping_results = []
        other_results = []

        for r in results:
            url = r.get("url", "")
            title = r.get("title", "")

            if self._is_shopping_site(url, title):
                shopping_results.append(r)
            else:
                other_results.append(r)

        # 優先回傳購物網站，不夠再補其他
        return shopping_results[:3] + other_results[: (3 - len(shopping_results))]

    async def _search_products(self, query: str, max_retries: int = 2, retry_delay: float = 2.0) -> list[dict[str, Any]]:
        """使用 Brave Search API 搜尋商品（帶完整錯誤處理）"""
        # 檢查快取
        cached = self.cache.get(query)
        if cached:
            return self._filter_search_results(cached)

        if not settings.BRAVE_API_KEY:
            raise RuntimeError("BRAVE_API_KEY 未設定，請在 .env 設定環境變數")

        last_error = None

        # 外層重試：處理 API 層級的連線問題
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.get(
                        BRAVE_SEARCH_URL,
                        headers={
                            "Accept": "application/json",
                            "Accept-Encoding": "gzip",
                            "X-Subscription-Token": settings.BRAVE_API_KEY,
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        },
                        params={
                            "q": query,
                            "count": settings.SEARCH_RESULTS_COUNT,
                            "ui_lang": "zh-TW",
                            "country": "tw",
                            "safesearch": "moderate",
                        },
                    )

                    # 檢查 HTTP 狀態碼
                    if resp.status_code == 401:
                        raise RuntimeError("Brave API 金鑰無效，請檢查設定")
                    elif resp.status_code == 403:
                        raise RuntimeError("Brave API 額度已用完或被封鎖")
                    elif resp.status_code == 429:
                        raise RuntimeError(f"Brave API 限流中，請稍後再試")
                    elif not resp.status_code == 200:
                        raise RuntimeError(f"Brave API 回應錯誤 ({resp.status_code}): {resp.text[:200]}")

                    data = resp.json()

                # 檢查回傳資料格式
                web_results = data.get("web", {}).get("results", [])
                if not web_results:
                    # 如果沒有結果，回傳空列表而不是拋出錯誤
                    print(f"  [警告] Brave Search 未回傳任何結果，請確認查詢字串")
                    return []

                results = [
                    {
                        "title": r.get("title", "（無標題）"),
                        "url": r.get("url", ""),
                        "description": r.get("description", ""),
                        "position": i + 1,
                    }
                    for i, r in enumerate(web_results)
                    if r.get("url")  # 過濾沒有 URL 的項目
                ]

                # 過濾結果
                filtered_results = self._filter_search_results(results)

                self.cache.set(query, filtered_results)
                await db.save_search_log(query, len(filtered_results))
                return filtered_results

            except httpx.HTTPStatusError as e:
                last_error = f"Brave API HTTP 錯誤 ({e.response.status_code})"
                print(f"  [API 錯誤] {last_error}")
                if attempt < max_retries:
                    print(f"  [重試] 等待 {retry_delay} 秒後重試...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指數退避

            except httpx.RequestError as e:
                last_error = f"Brave API 連線失敗: {type(e).__name__}"
                print(f"  [連線錯誤] {last_error}")
                if attempt < max_retries:
                    print(f"  [重試] 等待 {retry_delay} 秒後重試...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

            except json.JSONDecodeError as e:
                last_error = f"Brave API 回應格式錯誤: {e}"
                print(f"  [解析錯誤] {last_error}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

            except Exception as e:
                last_error = f"未預期錯誤: {type(e).__name__}: {str(e)}"
                print(f"  [錯誤] {last_error}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

        # 所有重試都失敗
        print(f"  [API 永久失敗] {last_error}")
        return []

    async def _scrape_with_retry(
        self,
        url: str,
        max_retries: int = 2,
        retry_delay: float = 3.0
    ) -> Optional[dict[str, Any]]:
        """帶重試機制的網頁抓取（增強版）"""
        for attempt in range(max_retries + 1):
            try:
                scrape_result = await self.scrape_agent.scrape(url)

                # 檢查抓取結果是否有效
                if scrape_result.markdown and len(scrape_result.markdown) > 50:
                    content = scrape_result.markdown
                    # 如果內容很長，取前 5000 + 後 5000（涵蓋價格資訊）
                    if len(content) > 10000:
                        content = content[:5000] + "\n[...content skipped...]\n" + content[-5000:]
                    return {
                        "url": url,
                        "markdown": content,
                        "title": scrape_result.title or "（無標題）",
                    }

            except asyncio.TimeoutError:
                print(f"  [抓取逾時] {url[:50]}... (嘗試 {attempt + 1}/{max_retries + 1})")

            except Exception as e:
                error_type = type(e).__name__
                print(f"  [抓取錯誤] {url[:50]}... {error_type}: {str(e)[:50]} (嘗試 {attempt + 1}/{max_retries + 1})")

            # 如果不是最後一次嘗試，等待後重試
            if attempt < max_retries:
                wait_time = retry_delay * (2 ** attempt) + (hash(url) % 2)  # 加入隨機抖動
                print(f"  [等待] 等待 {wait_time:.1f} 秒後重試...")
                await asyncio.sleep(wait_time)

        print(f"  [抓取永久失敗] {url}")
        return None

    async def _scrape_and_analyze(self, url: str, title: str = "") -> dict[str, Any]:
        """抓取網頁並使用 LLM 分析風險（增強版）"""
        # 嘗試抓取網頁
        scrape_data = await self._scrape_with_retry(url)

        if scrape_data is None:
            # 抓取失敗時返回預設值
            return {
                "url": url,
                "markdown": "",
                "price": 0,
                "has_suspicious_text": False,
                "risk_level": "medium",
                "title": title,
                "scrape_failed": True,
            }

        content = scrape_data["markdown"]
        page_title = scrape_data.get("title", "")

        # 提取價格（使用完整內容）
        initial_price = self._extract_price_from_text(content + " " + page_title)

        # 如果價格抓取不準確，使用 LLM 重新解析
        try:
            price, confidence = await asyncio.wait_for(
                self._extract_price_with_llm(content, page_title, initial_price),
                timeout=30.0  # 增加 LLM 超時時間
            )
        except asyncio.TimeoutError:
            print(f"  [LLM 價格提取逾時] 改用初始價格 {initial_price}")
            price = initial_price
            confidence = 0

        # 掃描詐騙關鍵字
        suspicious_keywords = self.price_guard.detect_suspicious_text(content)

        # 呼叫 Ollama 進行價格與風險判斷
        try:
            price_analysis = await asyncio.wait_for(
                self.price_guard.analyze_with_llm(content, price),
                timeout=45.0,  # 增加 LLM 超時時間
            )
            risk_level = price_analysis.risk_level
            # 如果 LLM 分析未包含 confidence，使用提取時的 confidence
            if price_analysis.price_confidence == 0:
                price_analysis.price_confidence = confidence
        except asyncio.TimeoutError:
            print(f"  [LLM 逾時] 改用規則判斷")
            price_analysis = self.price_guard.check_price_risk(price)
            price_analysis.price_confidence = confidence
            risk_level = price_analysis.risk_level
        except Exception as e:
            print(f"  [LLM 分析錯誤] {type(e).__name__}: {str(e)[:100]}，改用規則判斷")
            price_analysis = self.price_guard.check_price_risk(price)
            price_analysis.price_confidence = confidence
            risk_level = price_analysis.risk_level

        return {
            "url": url,
            "markdown": content,
            "price": price,
            "price_confidence": confidence,
            "has_suspicious_text": len(suspicious_keywords) > 0,
            "risk_level": risk_level,
            "title": title,
            "scrape_failed": False,
            "suspicious_keywords": suspicious_keywords,
        }

    async def compare(self, query: str) -> ComparisonResult:
        """執行完整比價流程（增強版錯誤處理）"""
        try:
            # 1. 搜尋商品
            try:
                search_results = await self._search_products(query)
            except RuntimeError as e:
                # Brave API 完全失敗，使用本地預設
                print(f"  [搜尋失敗] {e}")
                print("  [備用方案] 使用本地預設搜尋結果")
                search_results = []

            if not search_results:
                # 沒有任何搜尋結果，回傳空結果
                return ComparisonResult(
                    product_name=query,
                    recommendations=[],
                    cheapest=None,
                    safest=None,
                    most_recommended=None,
                    llm_recommendation="搜尋失敗或無可用商品資料，無法提供建議。"
                )

            # 1.5 使用 LLM 對搜尋結果進行評分與排序（只取前 3 名最相關的結果）
            try:
                ranked_results = await asyncio.wait_for(
                    self._rank_results_with_llm(query, search_results),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                print(f"  [LLM 排序] 超時，使用原始搜尋結果")
                ranked_results = search_results[:3]
            except Exception as e:
                print(f"  [LLM 排序] 錯誤: {type(e).__name__}，使用原始搜尋結果")
                ranked_results = search_results[:3]

            # 2. 抓取與分析每個結果（只針對 LLM 篩選後的前 3 名）
            # 使用 Semaphore 限制併發數量，避免 OOM
            semaphore = asyncio.Semaphore(3)

            async def _constrained_scrape(result):
                async with semaphore:
                    return await self._scrape_and_analyze(result["url"], result["title"])

            tasks = [_constrained_scrape(result) for result in ranked_results]
            analyses = await asyncio.gather(*tasks, return_exceptions=True)

            # 處理可能的錯誤
            processed_analyses = []
            for i, analysis in enumerate(analyses):
                if isinstance(analysis, Exception):
                    print(f"  [分析錯誤] 第 {i + 1} 個網址: {type(analysis).__name__}")
                    processed_analyses.append({
                        "url": search_results[i]["url"],
                        "markdown": "",
                        "price": 0,
                        "has_suspicious_text": False,
                        "risk_level": "medium",
                        "title": search_results[i].get("title", ""),
                        "scrape_failed": True,
                    })
                else:
                    processed_analyses.append(analysis)

            # 3. 組合比較結果
            products = []
            for i, (result, analysis) in enumerate(zip(search_results, processed_analyses)):
                # 如果網頁內容沒解析到價格，從搜尋結果描述找
                price = analysis.get("price", 0)
                if price == 0:
                    price = self._extract_price_from_text(
                        result.get("description", "")
                    )

                # 計算評分
                safety_score = ScoringEngine.calculate_safety_score(
                    analysis.get("risk_level", "low"),
                    analysis.get("has_suspicious_text", False),
                )
                price_score = ScoringEngine.calculate_price_score(price)

                products.append(
                    {
                        "name": result["title"],
                        "price": price,
                        "url": result["url"],
                        "source": self._extract_source(result["url"]),
                        "risk_level": analysis.get("risk_level", "low"),
                        "has_suspicious_text": analysis.get("has_suspicious_text", False),
                        "safety_score": safety_score,
                        "price_score": price_score,
                        "description": result["description"],
                        "scrape_failed": analysis.get("scrape_failed", False),
                        "suspicious_keywords": analysis.get("suspicious_keywords", []),
                    }
                )

            # 4. 分析並找出推薦選項
            if not products:
                return ComparisonResult(
                    product_name=query,
                    recommendations=[],
                    cheapest=None,
                    safest=None,
                    most_recommended=None,
                    llm_recommendation="所有商品抓取皆失敗，無法提供建議。"
                )

            # 過濾掉抓取失敗的商品（除非沒有其他選擇）
            successful_products = [p for p in products if not p["scrape_failed"]]
            candidates = successful_products if successful_products else products

            cheapest = min(candidates, key=lambda x: x["price"]) if candidates else None
            safest = max(candidates, key=lambda x: x["safety_score"]) if candidates else None

            # 最推薦：安全分數 > 60 且價格分數 > 40 的商品中，價格最低的
            recommended_candidates = [
                p
                for p in candidates
                if p["safety_score"] >= 60 and p["price_score"] >= 40
            ]
            most_recommended = (
                min(recommended_candidates, key=lambda x: x["price"])
                if recommended_candidates
                else (candidates[0] if candidates else None)
            )

            # 5. 儲存結果
            for product in products:
                try:
                    await db.save_shopping_result(
                        query=query,
                        product_name=product["name"],
                        product_url=product["url"],
                        source_platform=product["source"],
                        price=product["price"],
                        risk_level=product["risk_level"],
                        is_recommendation=(product == most_recommended),
                    )
                except Exception as e:
                    print(f"  [儲存錯誤] {e}")

            # 6. 使用 LLM 生成商品建議
            llm_recommendation = await self._generate_llm_recommendation(
                query, products
            )

            return ComparisonResult(
                product_name=query,
                recommendations=[
                    {
                        "name": p["name"],
                        "price": p["price"],
                        "url": p["url"],
                        "source": p["source"],
                        "risk_level": p["risk_level"],
                        "safety_score": p["safety_score"],
                        "price_score": p["price_score"],
                        "scrape_failed": p["scrape_failed"],
                        "suspicious_keywords": p["suspicious_keywords"],
                    }
                    for p in products
                ],
                cheapest=cheapest,
                safest=safest,
                most_recommended=most_recommended,
                llm_recommendation=llm_recommendation,
            )
        finally:
            # 確保 Crawl4AI / Playwright 相關資源被正確關閉
            try:
                await self.scrape_agent.__aexit__(None, None, None)
            except Exception:
                pass  # 忽略關閉時的錯誤

    async def _generate_llm_recommendation(
        self, query: str, products: list[dict[str, Any]]
    ) -> str:
        """使用 Ollama 生成商品購買建議（增強版）"""
        if not products:
            return "無法提供建議，沒有可用的商品資訊。"

        # 準備商品資訊（只取前 3 個避免超長）
        product_info = "\n".join(
            [
                f"商品{i+1}: {p['name'][:60]}... 價格: NT${p['price']:,} 安全分數: {p['safety_score']:.0f}"
                for i, p in enumerate(products[:3])
            ]
        )

        prompt = f"""你是一位專業的購物顧問。請根據以下商品資訊，為使用者提供購買建議。

商品關鍵字: {query}

商品比較:
{product_info}

請依照以下格式回應（使用繁體中文，總長度控制在 80 字以內）：
1. 總體建議：簡短總結哪個商品最值得購買
2. 風險提醒：如果價格過低或太高，請提醒使用者
3. 購買建議：是否建議立即購買

請直接給出建議，不要加入額外說明。"""

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt), timeout=30.0
            )
            recommendation = response.content.strip()
            # 確保長度合理
            return recommendation[:200] if len(recommendation) > 200 else recommendation
        except asyncio.TimeoutError:
            return "LLM 分析逾時，建議根據安全分數和價格自行評估。"
        except Exception as e:
            # 如果 LLM 失敗，使用規則生成建議
            print(f"  [LLM 生成建議錯誤] {type(e).__name__}: {str(e)[:50]}")
            if recommended := next((p for p in products if p["safety_score"] >= 60), None):
                return f"推薦選擇「{recommended['name'][:30]}...」，安全分數{recommended['safety_score']:.0f}分。"
            return "建議謹慎評估，目前商品的安全分數較低。"

    def _extract_price_from_text(self, text: str) -> float:
        """從文本提取價格，支援台灣電商常見格式（增強版）"""
        patterns = [
            (r"NT\$[\s]*([\d,]+)", "NT$"),
            (r"NTD[\s]*([\d,]+)", "NTD"),
            (r"價格[：:\s]*\$?[\s]*([\d,]+)", "價格"),
            (r"售價[：:\s]*\$?[\s]*([\d,]+)", "售價"),
            (r"優惠價[：:\s]*\$?[\s]*([\d,]+)", "優惠價"),
            (r"特價[：:\s]*\$?[\s]*([\d,]+)", "特價"),
            (r"特惠價[：:\s]*\$?[\s]*([\d,]+)", "特惠價"),
            (r"現價[：:\s]*\$?[\s]*([\d,]+)", "現價"),
            (r"定價[：:\s]*\$?[\s]*([\d,]+)", "定價"),
            (r"\$[\s]*([\d,]+)", "$"),
            (r"([\d,]+)\s*元", "元"),
            (r"([\d,]+)\s*NTD", "NTD"),
        ]

        best_price = 0.0
        best_line_score = 0

        for pattern, label in patterns:
            for line in text.splitlines():
                line = line.strip()
                # 避開常見的折扣促銷、分類選單字眼，避免誤抓
                if (
                    ("滿" in line and "折" in line)
                    or "以下" in line
                    or "以上" in line
                    or " - " in line
                    or "限時" in line
                    or "單品" in line
                    or "特惠" in line
                ):
                    continue

                # 搜尋所有匹配的價格（處理範圍价格如 NT$13,500 ~ NT$15,500）
                matches = list(re.finditer(pattern, line, re.IGNORECASE))
                for m in matches:
                    try:
                        price = float(m.group(1).replace(",", ""))
                        # 合理價格範圍：100 - 5,000,000
                        if 1000 <= price <= 5_000_000:
                            # 計算行分數（優先選擇包含"價格"、"售價"等關鍵字的行）
                            line_score = 0
                            if "價格" in line:
                                line_score += 10
                            if "售價" in line:
                                line_score += 10
                            if "現價" in line:
                                line_score += 5
                            if "特價" in line:
                                line_score += 3
                            # 價格越高，分數越高（假設昂貴商品更可能是目標）
                            line_score += min(price / 10000, 10)

                            if line_score > best_line_score:
                                best_line_score = line_score
                                best_price = price
                    except ValueError:
                        continue

        return best_price

    def _extract_source(self, url: str) -> str:
        """從網址提取來源平台"""
        u = url.lower()

        if "momo" in u:
            return "momo"
        if "shopee" in u:
            return "shopee"
        if "pchome" in u:
            return "pchome"
        if "yahoo" in u:
            return "yahoo"
        if "rakuten" in u:
            return "rakuten"
        if "books.com" in u:
            return "博客來"
        if "taobao" in u or "tmall" in u:
            return "淘寶"
        if "amazon" in u:
            return "amazon"
        if "lshopee" in u or "log shopee" in u:
            return "shopee"

        # 檢查網址是否包含 shopee 的各种可能形式
        if "shp" in u or "shopee" in u or "s.hpe" in u:
            return "shopee"

        return "other"

    async def _extract_price_with_llm(
        self, content: str, title: str, initial_price: float
    ) -> tuple[float, int]:
        """使用 LLM 重新解析價格（當爬蟲容易誤判時）

        使用 LangChain 的 with_structured_output 功能，
        結合 Pydantic 模型確保輸出格式正確。

        Returns:
            tuple: (price, confidence) - 價格與信心度 (0-100)
        """
        # 如果初步抓取的價格已經合理，直接返回
        if 1000 <= initial_price <= 1000000:
            return (initial_price, 80)  # 初步抓取的信心度為 80

        # 僅傳遞後 6000 字，因為價格資訊通常在頁面後段
        price_context = content[-6000:] if len(content) > 6000 else content

        # 構建提示模板
        prompt = f"""你是一位專業的價格解析員。請從以下網頁內容中找出商品的實際售價。

網頁標題：{title}

網頁內容（價格資訊在後段）：
{price_context}
"""

        try:
            # 使用 LangChain 的 with_structured_output 功能
            # 結合 PriceExtraction Pydantic 模型
            structured_llm = self.llm.with_structured_output(PriceExtraction)
            response = await asyncio.wait_for(
                structured_llm.ainvoke(prompt), timeout=15.0
            )

            # 直接從 Pydantic 物件取得結果，無需手動解析 JSON
            price = response.real_price
            confidence = response.confidence

            # 檢查信心度
            if confidence < 50:
                print(f"  [LLM 價格提取失敗] 信心度低於 50%: {confidence}")
                return (initial_price, 0)  # 信心度低，視為抓取失敗

            # 驗證價格範圍
            if 1000 <= price <= 5_000_000:
                print(f"  [LLM 修正價格] {initial_price} -> {price}, confidence: {confidence}")
                return (price, confidence)
            else:
                print(f"  [LLM 價格超出範圍] {price}，使用初始價格 {initial_price}")
                return (initial_price, 0)

        except Exception as e:
            print(f"  [LLM 價格解析錯誤] {type(e).__name__}: {str(e)[:50]}")

        # 如果 LLM 失敗，返回初始價格
        return (initial_price, 0)

    async def _rank_results_with_llm(
        self, query: str, search_results: list[dict]
    ) -> list[dict]:
        """使用 LLM 對搜尋結果進行評分與排序

        此函數在抓取網頁前，先讓 LLM 判斷哪些搜尋結果最可能是真實商品頁面，
        以避免浪費資源抓取無關網頁（如評測、開箱文等）。

        Args:
            query: 使用者的搜尋關鍵字
            search_results: Brave Search 的搜尋結果列表

        Returns:
            排序後的搜尋結果列表（只保留前 3 名最相關的結果）
        """
        # 如果搜尋結果少於等於 3 個，直接返回
        if len(search_results) <= 3:
            return search_results

        # 準備搜尋結果資訊
        results_info = "\n".join(
            [
                f"結果 {i+1}:\n"
                f"  標題: {r.get('title', 'N/A')}\n"
                f"  網址: {r.get('url', 'N/A')}\n"
                f"  描述: {r.get('description', 'N/A')[:150]}..."
                for i, r in enumerate(search_results[:10])  # 只送前 10 個給 LLM
            ]
        )

        # 構建提示模板
        prompt = f"""你是一位專業的購物搜尋專家。請根據以下搜尋結果，判斷哪些最可能是「真實商品販售頁面」。

搜尋關鍵字: {query}

搜尋結果：
{results_info}
"""

        try:
            # 設定 15 秒超時
            # 使用 LangChain 的 with_structured_output 功能
            # 結合 RankedURLs Pydantic 模型
            structured_llm = self.llm.with_structured_output(RankedURLs)
            response = await asyncio.wait_for(
                structured_llm.ainvoke(prompt), timeout=15.0
            )

            # 直接從 Pydantic 物件取得結果，無需手動解析 JSON
            ranked_urls = response.ranked_urls

            # 只保留前 3 個網址
            ranked_urls = ranked_urls[:3]

            if ranked_urls:
                # 根據 LLM 排序過濾搜尋結果
                ranked_results = [
                    r for r in search_results
                    if r.get("url") in ranked_urls
                ]

                # 如果排序結果不足 3 個，補上剩餘結果
                if len(ranked_results) < 3:
                    for r in search_results:
                        if r not in ranked_results:
                            ranked_results.append(r)
                            if len(ranked_results) >= 3:
                                break

                print(f"  [LLM 排序] 已篩選並排序 {len(ranked_results)} 個相關結果")
                return ranked_results

        except asyncio.TimeoutError:
            print(f"  [LLM 排序] 超時（15秒），使用原始順序")
        except Exception as e:
            print(f"  [LLM 排序] 錯誤: {type(e).__name__}: {str(e)[:50]}")

        # 防呆：如果 LLM 失敗，回退到原始順序的前 3 個結果
        print(f"  [防呆] 回退到原始搜尋結果的前 3 個")
        return search_results[:3]


async def perform_comparison(query: str) -> ComparisonResult:
    """快速執行比價的輔助函數"""
    async with ComparisonAgent() as agent:
        return await agent.compare(query)
