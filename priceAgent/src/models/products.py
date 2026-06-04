"""
================================================================================
產品與比價相關的 Pydantic 資料模型模組
================================================================================

本模組定義了整個比價機器人所使用的所有資料模型（Data Models），
使用 Pydantic V2 進行資料驗證與序列化。

主要資料模型：
1. Product - 商品基本資料
2. SearchResult - 搜尋引擎結果
3. ScrapeResult - 網頁抓取結果
4. PriceAnalysis - 價格分析結果
5. ComparisonResult - 比價比較結果
6. FraudDetectionResult - 防偽檢測結果
7. PriceGuardOutput - 價格守門員輸出

==============================================================================

資料模型設計原則：
1. 使用 Pydantic BaseModel 確保資料驗證
2. 所有欄位都有類型提示（Type Hinting）
3. 使用 Field() 提供詳細描述與預設值
4. 使用 model_validator 進行跨欄位驗證

==============================================================================
"""

# ==============================================================================
# 引入必要的外部套件
# ==============================================================================

import re  # 正規表達式，用於文字模式匹配
from datetime import datetime  # 日期時間處理
from decimal import Decimal  # 高精度小數計算（目前未使用，保留擴充性）
from typing import Any, Literal, Optional  # 類型提示工具

# Pydantic V2 的核心功能
# BaseModel: 資料模型基底類別
# Field: 欄位定義與驗證
# field_validator: 欄位層級驗證器
# model_validator: 模型層級驗證器
from pydantic import BaseModel, Field, field_validator, model_validator

# ==============================================================================
# 商品資料模型
# ==============================================================================


class Product(BaseModel):
    """
    商品資料模型 - 代表一個可購買的商品項目

    用途：
    ---------
    用於儲存和傳遞商品的基本資訊，當比價流程完成後，
    每個推薦商品都會轉換成此模型的實例。

    欄位說明：
    ---------
    1. name (str, 必填)：
       - 商品名稱
       - 例如："iPhone 15 128GB 黑色"

    2. price (float, 必填)：
       - 商品售價（新台幣）
       - 例如：32900.0
       - 所有價格都會轉換為浮點數以利比較

    3. url (str, 必填)：
       - 商品購買網址
       - 例如："https://momo.tw/iphone15"
       - 用於使用者點擊購買

    4. source (str, 必填)：
       - 來源平台識別碼
       - 可能值："momo"、"shopee"、"pchome"、"apple" 等
       - 用於識別商品來自哪個購物平台

    5. original_price (Optional[float])：
       - 原價（未折扣前）
       - 例如：35900.0
       - None 表示無原價資訊

    6. discount_rate (Optional[float])：
       - 折扣率（0.0 - 1.0）
       - 例如：0.9 表示 9 折
       - None 表示無折扣資訊

    使用範例：
    ---------
    product = Product(
        name="iPhone 15 128GB 黑色",
        price=32900.0,
        url="https://momo.tw/iphone15",
        source="momo",
        original_price=35900.0,
        discount_rate=0.9
    )
    print(product.name)  # 輸出："iPhone 15 128GB 黑色"
    print(product.price)  # 輸出：32900.0
    """

    name: str = Field(..., description="商品名稱")
    price: float = Field(..., description="商品價格")
    url: str = Field(..., description="商品網址")
    source: str = Field(..., description="來源平台（如：momo、Shopee）")
    original_price: Optional[float] = Field(None, description="原價")
    discount_rate: Optional[float] = Field(None, description="折扣率")


# ==============================================================================
# 搜尋結果資料模型
# ==============================================================================


class SearchResult(BaseModel):
    """
    搜尋結果資料模型 - 代表 Brave Search API 回傳的一筆搜尋結果

    用途：
    ---------
    用於儲存從搜尋引擎返回的商品候選頁面，
    ComparisonAgent 會根據這些結果進行篩選與抓取。

    欄位說明：
    ---------
    1. title (str)：
       - 搜尋結果的標題
       - 例如："iPhone 15 官方旗艦店 - 蘋果 (Apple) 台灣"

    2. url (str)：
       - 網頁的完整 URL
       - 例如："https://www.apple.com/tw/iphone-15/"

    3. description (str)：
       - 搜尋結果的描述文字
       - 通常包含商品簡介或關鍵資訊

    4. position (int, 必填)：
       - 搜尋結果的排名位置（從 1 開始）
       - 用途：排名越前表示相關性越高
       - 例如：position=1 表示第一個結果

    使用範例：
    ---------
    result = SearchResult(
        title="iPhone 15 官方旗艦店",
        url="https://apple.com/tw/iphone15",
        description="最新款 iPhone 15，配備 A17 處理器。",
        position=1
    )
    """

    title: str
    url: str
    description: str
    position: int = Field(..., description="搜尋結果排名")


# ==============================================================================
# 網頁抓取結果資料模型
# ==============================================================================


class ScrapeResult(BaseModel):
    """
    網頁抓取結果資料模型 - 代表從單一網頁抓取的完整內容

    用途：
    ---------
    ScrapeAgent 抓取網頁後會回傳此模型，
    包含頁面的原始 Markdown 內容、標題、圖片、連結等資訊。

    欄位說明：
    ---------
    1. url (str)：
       - 抓取來源的網址

    2. markdown (str)：
       - 網頁內容轉換成的 Markdown 格式
       - 用途：提供給 LLM 進行價格提取與分析
       - 僅包含純文字內容，去除廣告與不需要的元素

    3. title (Optional[str])：
       - 網頁標題
       - None 表示無法取得標題

    4. images (list[str])：
       - 網頁中所有圖片的 URL 清單
       - 預設為空列表 []

    5. links (list[str])：
       - 網頁中所有外部連結的 URL 清單
       - 預設為空列表 []

    6. scraped_at (str)：
       - 抓取時間（ISO 8601 格式）
       - 預設為當前時間
       - 例如："2024-01-15T10:30:00"

    使用範例：
    ---------
    result = ScrapeResult(
        url="https://momo.tw/iphone15",
        markdown="## iPhone 15\\n售價：NT$32,900\\n全新未拆封",
        title="iPhone 15 官方旗艦店",
        images=["https://example.com/img1.jpg"],
        links=["https://momo.tw/terms"]
    )
    """

    url: str
    markdown: str
    title: Optional[str] = None
    images: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    scraped_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ==============================================================================
# 價格分析資料模型
# ==============================================================================


class PriceAnalysis(BaseModel):
    """
    價格分析結果資料模型 - 代表 PriceGuardAgent 對單一商品價格的分析

    用途：
    ---------
    PriceGuardAgent 會分析網頁抓取到的價格，
    判斷價格是否合理、是否有詐騙風險，
    並提供購買建議與信心度。

    欄位說明：
    ---------
    1. is_valid (bool, 必填)：
       - 價格是否合理
       - True：價格在可接受範圍內
       - False：價格有問題（過低、過高或可疑）

    2. price (float, 必填)：
       - 分析後的價格（新台幣）
       - 可能是從文字中提取的價格

    3. risk_level (Literal["low", "medium", "high"], 必填)：
       - 價格風險等級
       - "low"：低風險，價格合理
       - "medium"：中風險，需要謹慎評估
       - "high"：高風險，建議避免

    4. risk_reason (Optional[str])：
       - 風險原因說明
       - 例如："價格低於市場平均 50%"
       - None 表示無風險或風險原因不明

    5. is_suspicious (bool, 必填)：
       - 是否檢測到可疑特徵
       - True：檢測到高仿、拆機等可疑字樣
       - False：未檢測到可疑特徵

    6. purchase_reason (Optional[str])：
       - 購買理由與商品優勢
       - 例如："品質良好，價格合理，比官方店便宜 10%"
       - None 表示沒有明確的購買理由

    7. price_confidence (int, 預設 0)：
       - 價格真實性信心度（0-100）
       - 來源：LLM 提取價格時的置信度
       - 0：未分析或信心度極低
       - 100：極高信心度

    使用範例：
    ---------
    analysis = PriceAnalysis(
        is_valid=True,
        price=32900.0,
        risk_level="low",
        is_suspicious=False,
        purchase_reason="品質良好，價格合理",
        price_confidence=90
    )
    """

    is_valid: bool = Field(..., description="價格是否合理")
    price: float = Field(..., description="分析後的價格")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="風險等級")
    risk_reason: Optional[str] = Field(None, description="風險原因")
    is_suspicious: bool = Field(..., description="是否可疑")
    purchase_reason: Optional[str] = Field(None, description="購買理由與商品優勢")
    price_confidence: int = Field(default=0, description="真實價格信心度 (0-100)")


# ==============================================================================
# 比價比較結果資料模型
# ==============================================================================


class ComparisonResult(BaseModel):
    """
    比價比較結果資料模型 - 代表一次比價流程的完整輸出

    用途：
    ---------
    ComparisonAgent.compare() 完成後會回傳此模型，
    包含所有商品建議、最便宜/最安全/最推薦選項，
    以及 LLM 生成的購買建議。

    欄位說明：
    ---------
    1. product_name (str)：
       - 搜尋的商品名稱
       - 例如："iPhone 15"

    2. recommendations (list[dict[str, Any]], 必填)：
       - 推薦選項列表（已排序）
       - 每個項目包含：名稱、價格、平台、風險等級、安全分數等
       - 按安全分數從高到低排序

    3. cheapest (Optional[dict[str, Any]])：
       - 最便宜的選項
       - 包含完整的商品資訊與價格
       - None 表示沒有可用的比較選項

    4. safest (Optional[dict[str, Any]])：
       - 最安全的選項（安全分數最高）
       - 包含完整的商品資訊與安全評估
       - None 表示沒有可用的比較選項

    5. most_recommended (Optional[dict[str, Any]])：
       - 最推薦的選項
       - 通常是安全分數最高的商品
       - None 表示沒有可用的比較選項

    6. llm_recommendation (Optional[str])：
       - Ollama LLM 生成的購買建議
       - 包含對所有選項的分析與推薦理由
       - None 表示沒有生成建議

    7. analyzed_at (str)：
       - 分析完成時間（ISO 8601 格式）
       - 預設為當前時間

    使用範例：
    ---------
    result = ComparisonResult(
        product_name="iPhone 15",
        recommendations=[...],
        cheapest=cheapest_item,
        safest=safest_item,
        most_recommended=recommended_item,
        llm_recommendation="建議選擇官方旗艦店，雖然較貴但品質有保障..."
    )
    """

    product_name: str
    recommendations: list[dict[str, Any]] = Field(
        ...,
        description="推薦選項列表"
    )
    cheapest: Optional[dict[str, Any]] = Field(None, description="最便宜選項")
    safest: Optional[dict[str, Any]] = Field(None, description="最安全選項")
    most_recommended: Optional[dict[str, Any]] = Field(None, description="最推薦選項")
    llm_recommendation: Optional[str] = Field(None, description="Ollama LLM 生成的購買建議")
    analyzed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ==============================================================================
# 防偽關鍵字
# ==============================================================================

# 可疑商品描述關鍵字清單
# 當網頁內容包含這些關鍵字時，會被標記為高風險
# 這些關鍵字通常表示商品可能是高仿、拆機、展示機等非全新商品
SUSPICIOUS_KEYWORDS: list[str] = [
    "高仿", "拆機", "僅供參考", "手機殼.*手機", "樣機", "展示機",
    "仿冒", "山寨", "A貨", "B貨", "C貨", "特供", "贈品", "瑕疵"
]

# 正規表達式編譯（用於高效匹配）
# _SUSPICIOUS_PATTERN = re.compile("|".join(SUSPICIOUS_KEYWORDS))


# ==============================================================================
# 防偽檢測結果資料模型
# ==============================================================================


class FraudDetectionResult(BaseModel):
    """
    防偽過濾結果資料模型 - 代表對單一商品的防偽檢測結果

    用途：
    ---------
    ScrapeAgent 抓取網頁後會進行防偽檢測，
    檢查是否包含可疑關鍵字，
    並與價格分析結果合併。

    欄位說明：
    ---------
    1. url (str)：
       - 檢測的商品網址

    2. has_suspicious_text (bool, 必填)：
       - 是否含有可疑文字
       - True：檢測到可疑關鍵字
       - False：未檢測到可疑關鍵字

    3. suspicious_keywords (list[str])：
       - 觸發的風險關鍵字清單
       - 例如：["高仿", "拆機"]
       - 預設為空列表 []

    4. price_analysis (Optional[PriceAnalysis])：
       - 附加的價格分析結果
       - None 表示尚未進行價格分析

    校驗規則（@model_validator）：
    ---------
    確保資料一致性：
    - 當 suspicious_keywords 非空時（有觸發關鍵字）
    - has_suspicious_text 必須為 True
    - 如果 not has_suspicious_text 但有關鍵字，會自動修正為 True

    使用範例：
    ---------
    result = FraudDetectionResult(
        url="https://momo.tw/iphone15",
        has_suspicious_text=True,
        suspicious_keywords=["高仿", "拆機"],
        price_analysis=price_analysis_result
    )
    """

    url: str
    has_suspicious_text: bool = Field(..., description="是否含有可疑文字")
    suspicious_keywords: list[str] = Field(default_factory=list, description="觸發的風險關鍵字")
    price_analysis: Optional[PriceAnalysis] = None

    @model_validator(mode="after")
    def validate_suspicious_text(self) -> "FraudDetectionResult":
        """
        模型層級驗證器 - 確保資料一致性

        校驗規則：
        ---------
        當 suspicious_keywords 清單非空時（檢測到關鍵字），
        has_suspicious_text 必須為 True。

        如果 not has_suspicious_text 但有關鍵字，
        自動修正 has_suspicious_text 為 True。

        返回值：
        ---------
        FraudDetectionResult：修正後的實例

        使用範例：
        ---------
        # 情況 1：有關鍵字但 has_suspicious_text=False
        # 會自動修正為 True
        result = FraudDetectionResult(
            url="...",
            has_suspicious_text=False,  # 錯誤的值
            suspicious_keywords=["高仿"]  # 但有關鍵字
        )
        # 修正後：result.has_suspicious_text == True

        # 情況 2：正常情况
        result = FraudDetectionResult(
            url="...",
            has_suspicious_text=True,
            suspicious_keywords=["高仿"]
        )
        # 保持不變
        """

        # 檢查是否有觸發的關鍵字
        if len(self.suspicious_keywords) > 0 and not self.has_suspicious_text:
            # 如果有關鍵字但 has_suspicious_text 是 False
            # 自動修正為 True，確保資料一致性
            return FraudDetectionResult(
                url=self.url,
                has_suspicious_text=True,
                suspicious_keywords=self.suspicious_keywords,
                price_analysis=self.price_analysis
            )

        # 資料一致，回傳原始實例
        return self


# ==============================================================================
# 價格守門員輸出資料模型
# ==============================================================================


class PriceGuardOutput(BaseModel):
    """
    價格守門員輸出資料模型 - PriceGuardAgent 的標準輸出格式

    用途：
    ---------
    PriceGuardAgent.analyze() 方法的標準回傳格式，
    包含原始價格、調整後價格、風險等級與警告訊息。

    欄位說明：
    ---------
    1. original_price (float)：
       - 原始提取的價格
       - 例如：32900.0

    2. adjusted_price (float)：
       - 調整後的價格
       - 可能根據折扣、優惠等進行調整

    3. risk_level (Literal["low", "medium", "high"])：
       - 風險等級
       - 與 PriceAnalysis 相同的分級機制

    4. warning_message (Optional[str])：
       - 警告訊息
       - 當價格有問題時提供詳細說明
       - 例如："價格低於市場平均 50%"
       - None 表示無警告

    使用範例：
    ---------
    output = PriceGuardOutput(
        original_price=32900.0,
        adjusted_price=32900.0,
        risk_level="low",
        warning_message=None
    )
    """

    original_price: float
    adjusted_price: float
    risk_level: Literal["low", "medium", "high"]
    warning_message: Optional[str] = None


# ==============================================================================
# LLM 結構化輸出資料模型
# ==============================================================================


class PriceExtraction(BaseModel):
    """
    LLM 價格提取結構化輸出模型
    用於 LangChain 的 with_structured_output 功能

    欄位說明：
    ---------
    real_price (float)：
      - 商品的實際售價（數字）
      - 如果找不到則為 0

    confidence (int)：
      - 對這個價格的信心度（0-100）
      - 基於價格是否在明顯的價格標示區域
    """

    real_price: float = Field(..., description="商品的實際售價")
    confidence: int = Field(..., description="對這個價格的信心度 (0-100)")


class RankedURLs(BaseModel):
    """
    LLM 搜尋結果排序結構化輸出模型
    用於 LangChain 的 with_structured_output 功能

    欄位說明：
    ---------
    ranked_urls (list[str])：
      - 排序後的網址清單（只包含最相關的 3 個商品頁面）
    """

    ranked_urls: list[str] = Field(
        ...,
        description="排序後的網址清單（只包含最相關的 3 個商品頁面）"
    )
