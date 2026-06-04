"""
================================================================================
價格守門員 Agent 模組
================================================================================

本模組提供價格風險檢查與詐騙特徵檢測的功能，
作為比價機器人的「價格守門員」，確保推薦的商品安全可靠。

主要功能：
1. PriceGuardAgent 類別 - 核心風險檢查器
   - detect_suspicious_text() - 掃描可疑關鍵字
   - check_price_risk() - 檢查價格風險
   - analyze_with_llm() - 使用 LLM 分析內容與價格
   - check_fraud() - 綜合檢查詐騙特徵

2. check_price_safety() - 輔助函數 - 快速檢查價格安全性

==============================================================================
"""

# ==============================================================================
# 引入必要的外部套件
# ==============================================================================

import asyncio  # 非同步操作，用於 LLM 調用超時控制
import re  # 正規表達式，用於匹配可疑關鍵字（仍用於可疑關鍵字掃描）
from typing import Any, Literal, Optional  # 類型提示工具
from pydantic import BaseModel, Field  # Pydantic 資料模型

# LangChain 核心套件
from langchain_core.language_models import BaseLLM  # LLM 基底類別

# 引入設定與資料模型
from ..core.config import settings  # 設定模組
from ..models.products import FraudDetectionResult, PriceAnalysis, SUSPICIOUS_KEYWORDS  # 資料模型


# ==============================================================================
# LLM 結構化輸出資料模型
# ==============================================================================


class LLMPriceAnalysisOutput(BaseModel):
    """
    LLM 價格分析結構化輸出模型
    用於 LangChain 的 with_structured_output 功能

    欄位說明：
    ---------
    is_valid (bool)：
      - 價格是否合理

    risk_level (Literal["low", "medium", "high"])：
      - 風險等級

    risk_reason (Optional[str])：
      - 風險原因說明

    is_suspicious (bool)：
      - 是否含有可疑特徵

    purchase_reason (Optional[str])：
      - 購買理由與商品優勢

    price_confidence (int)：
      - 價格真實性信心度 (0-100)
    """

    is_valid: bool
    risk_level: Literal["low", "medium", "high"]
    risk_reason: Optional[str] = None
    is_suspicious: bool
    purchase_reason: Optional[str] = None
    price_confidence: int = 0


# ==============================================================================
# 價格守門員 Agent 類別
# ==============================================================================


class PriceGuardAgent:
    """
    價格守門員 - 檢查價格與內容風險

    用途：
    ---------
    這個類別負責檢查比價結果中的價格與商品描述，
    判斷是否有風險或詐騙特徵，確保推薦給使用者的商品是安全的。

    風險檢查流程：
    ---------
    1. 價格檢查：
       - 檢查價格是否過低（低於最低門檻）
       - 檢查折扣是否過高（可能有詐騙）
       - 決定風險等級（low/medium/high）

    2. 內容檢查：
       - 掃描商品描述中的可疑關鍵字
       - 例如：高仿、拆機、展示機、樣機等
       - 判斷是否為非全新商品

    3. LLM 分析（可選）：
       - 如果有 LLM 實例，可以進行更深度的分析
       - 自動提取價格、判斷風險、生成購買建議

    使用範例：
    ---------
    from src.agents.price_guard import PriceGuardAgent

    agent = PriceGuardAgent()

    # 檢查價格風險
    analysis = agent.check_price_risk(price=32900)
    print(f"風險等級：{analysis.risk_level}")
    print(f"是否有效：{analysis.is_valid}")

    # 掃描可疑內容
    text = "這是一個高仿商品，拆機處理"
    keywords = agent.detect_suspicious_text(text)
    print(f"發現可疑關鍵字：{keywords}")

    # 綜合檢查
    fraud_result = agent.check_fraud(text=text, price=32900, original_price=35900)
    print(f"是否有可疑文字：{fraud_result.has_suspicious_text}")
    print(f"風險等級：{fraud_result.price_analysis.risk_level}")
    """

    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        初始化價格守門員 Agent

        參數說明：
        ---------
        llm (Optional[BaseLLM])：
          - 可選的 LLM 實例
          - 如果提供，可以進行深度的 LLM 分析
          - 如果為 None，只使用規則判斷

        使用範例：
        ---------
        # 不使用 LLM（只用規則）
        agent = PriceGuardAgent()

        # 使用 LLM（進行深度分析）
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="gemma3:27b")
        agent = PriceGuardAgent(llm=llm)
        """

        # 儲存 LLM 實例（可選）
        # 如果提供 LLM，可以進行深度的內容分析
        self.llm = llm

        # 編譯可疑關鍵字的正規表達式
        # 使用 re.IGNORECASE 讓匹配不區分大小寫
        # 這樣可以提高匹配的靈活性
        self.suspicious_patterns = [
            re.compile(kw, re.IGNORECASE)
            for kw in SUSPICIOUS_KEYWORDS
        ]

    # --------------------------------------------------------------------------
    # 可疑內容掃描
    # ------------------------------------------------------

    def detect_suspicious_text(self, text: str) -> list[str]:
        """
        掃描文本中的可疑關鍵字

        用途：
        ---------
        檢查商品描述中是否包含可疑關鍵字，
        例如：高仿、拆機、展示機、樣機等。

        匹配方式：
        ---------
        - 使用正規表達式進行匹配
        - 不區分大小寫（IGNORECASE）
        - 返回所有匹配的關鍵字

        參數說明：
        ---------
        text (str)：
          - 要檢查的文本（商品描述）
          - 可以是 Markdown 格式或其他文字內容

        返回值：
        ---------
        list[str]：找到的可疑關鍵字清單
          - []：沒有找到可疑關鍵字
          - ["高仿", "拆機"]：找到這些關鍵字

        使用範例：
        ---------
        # 情況 1：沒有可疑內容
        text = "這是一個全新的 Apple 產品，品質優良"
        keywords = agent.detect_suspicious_text(text)
        print(keywords)  # 輸出：[]

        # 情況 2：有可疑內容
        text = "這是一個高仿商品，拆機處理"
        keywords = agent.detect_suspicious_text(text)
        print(keywords)  # 輸出：["高仿", "拆機"]

        涉及的關鍵字：
        -------
        SUSPICIOUS_KEYWORDS = [
            "高仿", "拆機", "僅供參考", "手機殼.*手機", "樣機", "展示機",
            "仿冒", "山寨", "A貨", "B貨", "C貨", "特供", "贈品", "瑕疵"
        ]
        """

        # 初始化找到的關鍵字清單
        found_keywords = []

        # 遍歷所有編譯過的正規表達式模式
        for pattern in self.suspicious_patterns:
            # 使用 pattern.search() 檢查文本中是否包含該模式
            # 如果找到匹配，pattern.search() 會返回 Match 物件
            if pattern.search(text):
                # 將匹配到的關鍵字（pattern.pattern）加入清單
                found_keywords.append(pattern.pattern)

        # 返回找到的關鍵字清單
        return found_keywords

    # --------------------------------------------------------------------------
    # 價格風險檢查
    # ------------------------------------------------------

    def check_price_risk(
        self,
        price: float,
        original_price: Optional[float] = None
    ) -> PriceAnalysis:
        """
        檢查價格風險

        用途：
        ---------
        根據商品價格與設定的門檻，判斷價格風險等級。
        這是價格守門員的核心檢查邏輯。

        風險檢查規則：
        ---------
        1. 價格過低檢查：
           - 如果 price < MIN_PRICE_THRESHOLD * RISK_PRICE_MULTIPLIER
             → 風險等級：high，is_valid：False
             → 例如：最低門檻 100，倍數 0.5，低於 50 為高風險

           - 如果 price < MIN_PRICE_THRESHOLD
             → 風險等級：medium
             → 例如：低於 100 為中風險

           - 否則
             → 風險等級：low

        2. 折扣檢查（如果提供 original_price）：
           - 計算折扣率：discount_rate = (original - price) / original
           - 如果折扣率 > 0.9（9 折以上）
             → 風險等級提升為 medium
             → 加入折扣過高的警告訊息

        參數說明：
        ---------
        price (float)：
          - 商品售價（新台幣）

        original_price (Optional[float])：
          - 原價（未折扣前）
          - 如果提供，會額外檢查折扣是否過高

        返回值：
        ---------
        PriceAnalysis：價格分析結果物件
          - is_valid (bool)：價格是否有效
          - price (float)：檢查的價格
          - risk_level (str)："low" | "medium" | "high"
          - risk_reason (Optional[str])：風險原因說明
          - is_suspicious (bool)：是否可疑（中高風險為 True）

        使用範例：
        ---------
        # 情況 1：價格正常
        analysis = agent.check_price_risk(price=32900)
        print(analysis.risk_level)  # 輸出："low"

        # 情況 2：價格過低
        analysis = agent.check_price_risk(price=40)
        print(analysis.risk_level)  # 輸出："high"
        print(analysis.is_valid)    # 輸出：False

        # 情況 3：價格偏低
        analysis = agent.check_price_risk(price=80)
        print(analysis.risk_level)  # 輸出："medium"

        # 情況 4：折扣過高
        analysis = agent.check_price_risk(
            price=32900,
            original_price=35900
        )
        # 折扣率 = (35900-32900)/35900 = 0.083（8.3 折），不超過 9 折
        """

        # 初始化風險狀態
        risk_level = "low"  # 預設為低風險
        risk_reason = None  # 預設無風險原因
        is_valid = True     # 預設價格有效

        # --------------------------------------
        # 檢查價格是否過低
        # --------------------------------------

        # 計算高風險門檻
        high_risk_threshold = settings.MIN_PRICE_THRESHOLD * settings.RISK_PRICE_MULTIPLIER

        # 檢查價格是否低於高風險門檻
        if price < high_risk_threshold:
            # 高風險：價格極度偏低，極可能有詐騙
            risk_level = "high"
            is_valid = False
            risk_reason = f"價格過低（{price}），低於安全門檻（{high_risk_threshold}）"

        # 檢查價格是否低於最低門檻（但高於高風險門檻）
        elif price < settings.MIN_PRICE_THRESHOLD:
            # 中風險：價格偏低，需要謹慎評估
            risk_level = "medium"
            risk_reason = f"價格偏低（{price}），接近最低門檻（{settings.MIN_PRICE_THRESHOLD}）"

        # --------------------------------------
        # 檢查折扣是否過高（如果有原價）
        # --------------------------------------

        # 如果有提供原價，檢查折扣率
        if original_price and original_price > price:
            # 計算折扣率：(原價 - 售價) / 原價
            discount_rate = (original_price - price) / original_price

            # 如果折扣率超過 90%（9 折以上），提高風險等級
            if discount_rate > 0.9:
                # 提高風險等級為中風險
                risk_level = "medium"
                # 如果沒有風險原因，加上折扣過高的警告
                risk_reason = risk_reason or f"折扣過高（{discount_rate*100:.0f}%），需注意"

        # --------------------------------------
        # 構建並返回分析結果
        # --------------------------------------

        # 構建 PriceAnalysis 物件
        # is_suspicious：如果風險等級是 medium 或 high，則為 True
        return PriceAnalysis(
            is_valid=is_valid,
            price=price,
            risk_level=risk_level,
            risk_reason=risk_reason,
            is_suspicious=risk_level in ["medium", "high"]
        )

    # --------------------------------------------------------------------------
    # LLM 深度分析
    # ------------------------------------------------------

    async def analyze_with_llm(self, text: str, price: float) -> PriceAnalysis:
        """
        使用 LLM 分析內容與價格

        用途：
        ---------
        如果有 LLM 實例，可以進行深度的內容分析：
        - 自動提取價格資訊
        - 判斷內容是否有詐騙特徵
        - 生成購買建議與理由
        - 評估價格真實性信心度

        如果沒有 LLM，會回退到規則判斷（check_price_risk）

        分析流程：
        ---------
        1. 檢查是否有 LLM 實例
           - 沒有：回退到 check_price_risk() 使用規則判斷

        2. 使用 LangChain 的 with_structured_output
           - 結合 LLMPriceAnalysisOutput Pydantic 模型
           - 自動確保輸出格式正確

        3. 呼叫 LLM 進行分析
           - 使用 structured_llm.ainvoke() 進行非同步調用

        參數說明：
        ---------
        text (str)：
          - 商品描述文字
          - 最多取前 2000 字元（避免 Token 限制）

        price (float)：
          - 商品售價

        返回值：
        ---------
        PriceAnalysis：價格分析結果物件

        使用範例：
        ---------
        from langchain_ollama import ChatOllama
        from src.agents.price_guard import PriceGuardAgent

        llm = ChatOllama(model="gemma3:27b")
        agent = PriceGuardAgent(llm=llm)

        text = "iPhone 15 官方旗艦店，全新未拆封，台灣版"
        analysis = await agent.analyze_with_llm(text, 32900)

        print(f"價格有效：{analysis.is_valid}")
        print(f"風險等級：{analysis.risk_level}")
        print(f"購買理由：{analysis.purchase_reason}")
        print(f"信心度：{analysis.price_confidence}")
        """

        # 如果沒有 LLM 實例，回退到規則判斷
        if not self.llm:
            return self.check_price_risk(price)

        # 構建提示（簡化版，LangChain 會自動處理 Pydantic schema）
        prompt = f"""你是一個專業的價格審查員與購物顧問。請分析以下商品描述與價格，判斷風險並提供購買建議。

商品描述（前 2000 字）：
{text[:2000]}

價格：{price}

請根據商品描述與價格，提供以下資訊：
1. 價格是否合理
2. 是否含有可疑關鍵字（高仿、拆機、僅供參考、手機殼偽裝成手機等）
3. 風險等級：low/medium/high
4. 購買理由與商品優勢（如果有）

請以結構化方式回應，確保資訊準確。
"""

        try:
            # 使用 LangChain 的 with_structured_output 功能
            # 結合 LLMPriceAnalysisOutput Pydantic 模型
            structured_llm = self.llm.with_structured_output(LLMPriceAnalysisOutput)
            response = await asyncio.wait_for(
                structured_llm.ainvoke(prompt), timeout=30.0
            )

            # 直接從 Pydantic 物件取得結果，無需手動解析 JSON
            return PriceAnalysis(
                is_valid=response.is_valid,
                price=price,
                risk_level=response.risk_level,
                risk_reason=response.risk_reason,
                is_suspicious=response.is_suspicious,
                purchase_reason=response.purchase_reason,
                price_confidence=response.price_confidence
            )

        except asyncio.TimeoutError:
            print(f"  [LLM 分析逾時] 改用規則判斷")
            return self.check_price_risk(price)
        except Exception as e:
            print(f"  [LLM 分析錯誤] {type(e).__name__}: {str(e)[:50]}，改用規則判斷")
            return self.check_price_risk(price)

    def check_fraud(
        self,
        text: str,
        price: float,
        original_price: Optional[float] = None
    ) -> FraudDetectionResult:
        """
        綜合檢查詐騙特徵

        用途：
        ---------
        這是一個整合函數，同時檢查內容與價格的風險，
        回傳完整的詐騙檢測結果。

        檢查項目：
        ---------
        1. 內容檢查：
           - 掃描可疑關鍵字（detect_suspicious_text）

        2. 價格檢查：
           - 檢查價格風險（check_price_risk）

        參數說明：
        ---------
        text (str)：
          - 商品描述文字

        price (float)：
          - 商品售價

        original_price (Optional[float])：
          - 原價（用於折扣檢查）

        返回值：
        ---------
        FraudDetectionResult：詐騙檢測結果物件
          - url (str)：網址（預設為空字串）
          - has_suspicious_text (bool)：是否有可疑文字
          - suspicious_keywords (list[str])：可疑關鍵字清單
          - price_analysis (PriceAnalysis)：價格分析結果

        使用範例：
        ---------
        result = agent.check_fraud(
            text="iPhone 15 官方旗艦店，全新未拆封",
            price=32900,
            original_price=35900
        )

        print(f"是否有可疑文字：{result.has_suspicious_text}")
        print(f"可疑關鍵字：{result.suspicious_keywords}")
        print(f"價格風險：{result.price_analysis.risk_level}")
        """

        # 檢查可疑關鍵字
        suspicious_keywords = self.detect_suspicious_text(text)

        # 檢查價格風險
        price_analysis = self.check_price_risk(price, original_price)

        # 構建詐騙檢測結果
        return FraudDetectionResult(
            url="",  # 預設為空字串，實際使用時應填入網址
            has_suspicious_text=len(suspicious_keywords) > 0,
            suspicious_keywords=suspicious_keywords,
            price_analysis=price_analysis
        )


# ==============================================================================
# 輔助函數
# ==============================================================================

def check_price_safety(
    price: float,
    original_price: Optional[float] = None
) -> dict[str, Any]:
    """
    快速檢查價格安全性的輔助函數

    用途：
    ---------
    這是一個快速檢查價格安全性的便利函數，
    不需要建立 PriceGuardAgent 實例。

    使用方式：
    ---------
    result = check_price_safety(32900)
    print(result)

    返回值：
    ---------
    dict[str, Any]：價格安全性檢查結果
      - 使用 PriceAnalysis.model_dump() 轉換為字典

    使用範例：
    ---------
    # 情況 1：價格正常
    result = check_price_safety(32900)
    print(result["risk_level"])  # 輸出："low"
    print(result["is_valid"])    # 輸出：True

    # 情況 2：價格過低
    result = check_price_safety(40)
    print(result["risk_level"])  # 輸出："high"
    print(result["is_valid"])    # 輸出：False

    # 情況 3：帶原價檢查
    result = check_price_safety(32900, 35900)
    print(result["risk_level"])  # 輸出："low"
    """

    # 建立 PriceGuardAgent 實例（使用預設設定）
    agent = PriceGuardAgent()

    # 執行價格風險檢查
    analysis = agent.check_price_risk(price, original_price)

    # 將分析結果轉換為字典並返回
    return analysis.model_dump()
