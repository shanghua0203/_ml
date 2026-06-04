"""
================================================================================
比價评分引擎模組
================================================================================

本模組提供商品評分與風險分級的功能，
用於綜合評估比價結果的安全性與合理性。

主要功能：
1. calculate_safety_score() - 計算安全分數（0-100）
2. calculate_price_score() - 計算價格合理性分數（0-100）
3. calculate_overall_score() - 計算綜合分數（0-100）
4. classify_risk() - 分級價格風險
5. get_recommendation_rank() - 取得推薦等級

==============================================================================
"""

# ==============================================================================
# 引入必要的外部套件
# ==============================================================================

from decimal import Decimal  # 高精度小數計算（保留擴充性）
from typing import Any  # 類型提示工具

# ==============================================================================
# 評分引擎類別
# ==============================================================================


class ScoringEngine:
    """
    商品評分引擎 - 絼合評估價格、風險等因素

    用途：
    ---------
    當比價流程完成後，ScoringEngine 會為每個商品計算以下分數：
    1. 安全分數：評估商品是否可靠（防偽檢測結果）
    2. 價格分數：評估價格是否合理
    3. 綜合分數：安全 60% + 價格 40%

    這些分數用於排序與推薦，幫助使用者選擇最適合的商品。

    使用範例：
    ---------
    from src.utils.scoring import ScoringEngine

    # 計算商品的安全分數
    safety_score = ScoringEngine.calculate_safety_score(
        risk_level="low",
        has_suspicious_text=False
    )
    print(f"安全分數：{safety_score}")  # 輸出：100

    # 計算商品的價格分數
    price_score = ScoringEngine.calculate_price_score(
        price=32900,
        min_price=100,
        max_price=50000
    )
    print(f"價格分數：{price_score}")  # 輸出：90.0

    # 計算綜合分數
    product = {
        "risk_level": "low",
        "has_suspicious_text": False,
        "price": 32900
    }
    overall_score = ScoringEngine.calculate_overall_score(product)
    print(f"綜合分數：{overall_score}")  # 輸出：94.0
    """

    # --------------------------------------------------------------------------
    # 安全分數計算
    # --------------------------------------------------------------

    @staticmethod
    def calculate_safety_score(risk_level: str, has_suspicious_text: bool) -> float:
        """
        計算安全分數（0-100）

        用途：
        ---------
        根據商品的風險等級與是否含有可疑文字，
        計算商品的安全性分數。

        評分規則：
        ---------
        1. 基礎分數：100 分

        2. 風險等級扣分：
           - high（高風險）：扣 60 分 → 剩 40 分
           - medium（中風險）：扣 30 分 → 剩 70 分
           - low（低風險）：不扣分 → 剩 100 分

        3. 可疑文字額外扣分：
           - has_suspicious_text = True：額外扣 30 分
           - has_suspicious_text = False：不扣分

        4. 分數限制：
           - 最低 0 分，最高 100 分
           - 使用 max(0, min(100, base_score)) 確保範圍

        參數說明：
        ---------
        risk_level (str)：
          - "low"：低風險商品
          - "medium"：中風險商品
          - "high"：高風險商品

        has_suspicious_text (bool)：
          - True：檢測到可疑關鍵字
          - False：未檢測到可疑關鍵字

        返回值：
        ---------
        float：安全分數（0-100）

        範例計算：
        ---------
        # 情況 1：低風險 + 無可疑文字
        calculate_safety_score("low", False)
        = 100 - 0 - 0 = 100

        # 情況 2：高風險 + 有可疑文字
        calculate_safety_score("high", True)
        = 100 - 60 - 30 = 10

        # 情況 3：中風險 + 無可疑文字
        calculate_safety_score("medium", False)
        = 100 - 30 - 0 = 70
        """

        # 設定基礎分數為 100 分
        base_score = 100

        # 根據風險等級進行扣分
        if risk_level == "high":
            # 高風險：扣 60 分
            base_score -= 60
        elif risk_level == "medium":
            # 中風險：扣 30 分
            base_score -= 30
        # low 風險不扣分

        # 如果有可疑文字，額外扣 30 分
        if has_suspicious_text:
            base_score -= 30

        # 確保分數在 0-100 之間
        # max(0, ...) 確保不低於 0
        # min(100, ...) 確保不高於 100
        return max(0, min(100, base_score))

    # --------------------------------------------------------------------------
    # 價格分數計算
    # --------------------------------------------------------------

    @staticmethod
    def calculate_price_score(
        price: float,
        min_price: float = 100,
        max_price: float = 50000
    ) -> float:
        """
        計算價格合理性分數（0-100）

        用途：
        ---------
        根據商品價格與合理價格區間的關係，
        計算價格的合理性分數。

        評分規則：
        ---------
        1. 價格過低（< min_price）：
           - price < min_price * 0.5：20 分（極低價格）
           - min_price * 0.5 <= price < min_price：40-60 分
           - 越接近 min_price，分數越高

        2. 價格過高（> max_price）：
           - price > max_price * 2：20 分（極高價格）
           - max_price < price <= max_price * 2：60-80 分
           - 越接近 max_price，分數越高

        3. 價格合理（min_price <= price <= max_price）：
           - 以中位數為基準計算分數
           - 越接近中位數，分數越高
           - 最高 100 分，最低 50 分

        參數說明：
        ---------
        price (float)：
          - 商品售價（新台幣）

        min_price (float，預設 100)：
          - 合理價格區間的下限
          - 低於此價格會被標記為高風險

        max_price (float，預設 50000)：
          - 合理價格區間的上限
          - 高於此價格可能會被標記為異常

        返回值：
        ---------
        float：價格分數（0-100）

        範例計算：
        ---------
        # 情況 1：價格過低
        calculate_price_score(40, 100, 50000)
        = 20（低於 50 元，極低價格）

        # 情況 2：價格合理且接近中位數
        calculate_price_score(25000, 100, 50000)
        = 100（正好在中位數 25050）

        # 情況 3：價格合理但偏離中位數
        calculate_price_score(100, 100, 50000)
        = 85（接近下限，有一定扣分）
        """

        # 檢查價格是否低於最低門檻
        if price < min_price:
            # 過低的價格，按比例扣分
            if price < min_price * 0.5:
                # 低於 50% 最低門檻：極低價格，分數很低
                return 20.0
            # 介於 50%-100% 之間：按比例計算分數
            # 範圍：40-60 分
            return 40.0 + (price - min_price * 0.5) / (min_price * 0.5) * 20

        # 檢查價格是否高於最高門檻
        if price > max_price:
            # 過高的價格，按比例扣分
            if price > max_price * 2:
                # 高於 200% 最高門檻：極高價格，分數很低
                return 20.0
            # 介於 100%-200% 之間：按比例計算分數
            # 範圍：60-80 分
            return 60.0 + (max_price * 2 - price) / (max_price * 2 - max_price) * 20

        # 價格在合理範圍內，計算分數
        # 計算中位數
        mid_price = (min_price + max_price) / 2
        # 計算與中位數的距離
        distance_from_mid = abs(price - mid_price)
        # 計算最大可能距離（中位數到極端值）
        max_distance = (max_price - min_price) / 2

        # 距離中位數越遠，分數越低
        # 計算扣分：距離越遠，扣分越多
        score = 100 - (distance_from_mid / max_distance) * 30

        # 確保分數在 50-100 之間
        # 合理範圍內的最低分數為 50
        return max(50, min(100, score))

    # --------------------------------------------------------------------------
    # 綜合分數計算
    # --------------------------------------------------------------

    @classmethod
    def calculate_overall_score(cls, product: dict[str, Any]) -> float:
        """
        計算商品綜合分數（0-100）

        用途：
        ---------
        統合安全分數與價格分數，計算商品的最終評分。
        用於比價結果的排序與推薦。

        權重分配：
        ---------
        - 安全分數：60%（較重要，確保商品可靠性）
        - 價格分數：40%（考慮價格合理性）

        評分步驟：
        ---------
        1. 取得商品的風險等級與可疑文字狀態
        2. 計算安全分數（0-100）

        3. 取得商品價格
        4. 根據價格範圍選擇合適的評分區間：
           - < 1000 元：區間 100-5000
           - < 10000 元：區間 1000-50000
           - < 50000 元：區間 5000-200000
           - >= 50000 元：區間 10000-500000

        5. 計算價格分數（0-100）

        6. 綜合分數 = 安全分數 * 0.6 + 價格分數 * 0.4

        參數說明：
        ---------
        product (dict[str, Any])：
          - 商品資料字典
          - 必要欄位：
            - "risk_level"：風險等級（"low" | "medium" | "high"）
            - "has_suspicious_text"：是否有可疑文字（bool）
            - "price"：商品價格（float）

        返回值：
        ---------
        float：綜合分數（0-100）

        使用範例：
        ---------
        product = {
            "risk_level": "low",
            "has_suspicious_text": False,
            "price": 32900
        }

        score = ScoringEngine.calculate_overall_score(product)
        # 假設安全分數 = 100，價格分數 = 88
        # 綜合分數 = 100 * 0.6 + 88 * 0.4 = 95.2
        """

        # 計算安全分數
        safety_score = cls.calculate_safety_score(
            product.get("risk_level", "low"),  # 預設為 "low"
            product.get("has_suspicious_text", False),  # 預設為 False
        )

        # 取得商品價格
        price = product.get("price", 0)

        # 根據商品價格選擇合適的評分區間
        # 不同價格區間使用不同的評分標準
        if price < 1000:
            # < 1000 元：小型商品
            price_min, price_max = 100, 5000
        elif price < 10000:
            # < 10000 元：中型商品（3C 配件、小家電）
            price_min, price_max = 1000, 50000
        elif price < 50000:
            # < 50000 元：大型商品（筆電、手機）
            price_min, price_max = 5000, 200000
        else:
            # >= 50000 元：高價商品（高階筆電、平板）
            price_min, price_max = 10000, 500000

        # 計算價格分數
        price_score = cls.calculate_price_score(price, price_min, price_max)

        # 計算綜合分數
        # 權重：安全 60%，價格 40%
        return safety_score * 0.6 + price_score * 0.4

    # --------------------------------------------------------------------------
    # 價格風險分級
    # --------------------------------------------------------------

    @staticmethod
    def classify_risk(price: float, min_threshold: float = 100) -> str:
        """
        分級價格風險

        用途：
        ---------
        根據價格與最低門檻的關係，
        自動分級風險等級。

        風險分級：
        ---------
        - high（高風險）：price < min_threshold * 0.5
          價格低於門檻的一半，極度可疑

        - medium（中風險）：min_threshold * 0.5 <= price < min_threshold
          價格低於門檻，需要謹慎評估

        - low（低風險）：price >= min_threshold
          價格在正常範圍內

        參數說明：
        ---------
        price (float)：
          - 商品售價

        min_threshold (float，預設 100)：
          - 最低價格門檻

        返回值：
        ---------
        str：風險等級（"low" | "medium" | "high"）

        使用範例：
        ---------
        # 價格過低
        classify_risk(40, 100)  # "high"（低於 50）

        # 價格接近門檻
        classify_risk(80, 100)  # "medium"（介於 50-100）

        # 價格正常
        classify_risk(32900, 100)  # "low"（高於 100）
        """

        # 計算高風險的門檻（最低門檻的 50%）
        high_risk_threshold = min_threshold * 0.5

        # 分級判斷
        if price < high_risk_threshold:
            return "high"
        elif price < min_threshold:
            return "medium"
        return "low"

    # --------------------------------------------------------------------------
    # 推薦等級取得
    # --------------------------------------------------------------

    @staticmethod
    def get_recommendation_rank(score: float) -> str:
        """
        根據分數取得推薦等級

        用途：
        ---------
        根據商品的綜合分數，
        判斷推薦等級，用於使用者介面顯示。

        推薦等級：
        ---------
        - "recommended"（推薦）：score >= 80
          高品質商品，強烈推薦購買

        - "acceptable"（可接受）：score >= 60
          品質良好，可以考慮購買

        - "caution"（注意）：score >= 40
          有風險，需要謹慎評估

        - "avoid"（避免）：score < 40
          高風險商品，建議避免

        參數說明：
        ---------
        score (float)：
          - 商品綜合分數（0-100）

        返回值：
        ---------
        str：推薦等級（"recommended" | "acceptable" | "caution" | "avoid"）

        使用範例：
        ---------
        get_recommendation_rank(95)  # "recommended"
        get_recommendation_rank(75)  # "acceptable"
        get_recommendation_rank(50)  # "caution"
        get_recommendation_rank(30)  # "avoid"
        """

        # 根據分數範圍返回對應的推薦等級
        if score >= 80:
            return "recommended"
        elif score >= 60:
            return "acceptable"
        elif score >= 40:
            return "caution"
        return "avoid"
