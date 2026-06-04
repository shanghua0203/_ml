"""
tests/test_text_processor.py — 文字處理器單元測試（v1.3）
測試 min_freq 低頻詞過濾功能
"""

import pytest
from text_processor import (
    build_vocab,
    text_to_ids,
    tokenize,
    UNK_TOKEN,
    UNK_ID,
    FALLBACK_STORY,
    DOMAIN_WORDS,
)


class TestBuildVocabMinFreq:
    """測試 min_freq 低頻詞過濾"""

    def test_min_freq_1_keeps_all(self):
        text = "咖啡咖啡溫度森林"
        w2i, i2w, vs = build_vocab(text, min_freq=1)
        assert "咖啡" in w2i
        assert "溫度" in w2i
        assert "森林" in w2i
        assert vs == 4

    def test_min_freq_2_removes_singletons(self):
        text = "咖啡咖啡溫度森林"
        w2i, i2w, vs = build_vocab(text, min_freq=2)
        assert "咖啡" in w2i
        assert "溫度" not in w2i
        assert "森林" not in w2i
        assert vs == 2

    def test_min_freq_3_removes_low_freq(self):
        text = "咖啡咖啡咖啡溫度溫度森林"
        w2i, i2w, vs = build_vocab(text, min_freq=3)
        assert "咖啡" in w2i
        assert "溫度" not in w2i
        assert "森林" not in w2i
        assert vs == 2

    def test_unk_always_present(self):
        w2i, i2w, _ = build_vocab("", min_freq=3)
        assert UNK_TOKEN in w2i
        assert w2i[UNK_TOKEN] == UNK_ID
        assert UNK_ID in i2w
        assert i2w[UNK_ID] == UNK_TOKEN

    def test_empty_text_with_min_freq(self):
        w2i, i2w, vs = build_vocab("", min_freq=3)
        assert vs == 1
        assert len(w2i) == 1
        assert w2i[UNK_TOKEN] == UNK_ID

    def test_min_freq_high_removes_all_but_unk(self):
        text = "咖啡溫度森林"
        w2i, i2w, vs = build_vocab(text, min_freq=10)
        assert vs == 1
        assert len(w2i) == 1

    def test_default_min_freq_is_1(self):
        w2i, i2w, vs = build_vocab("咖啡溫度森林")
        assert vs == 4
        assert "咖啡" in w2i

    def test_id_to_word_consistent(self):
        w2i, i2w, vs = build_vocab("咖啡咖啡溫度森林", min_freq=2)
        for word, idx in w2i.items():
            if word != UNK_TOKEN:
                assert i2w[idx] == word
        assert i2w[UNK_ID] == UNK_TOKEN


class TestTextToIdsMinFreq:
    """測試低頻詞過濾後 text_to_ids 的映射正確性"""

    def test_low_freq_maps_to_unk(self):
        text = "咖啡咖啡溫度森林"
        w2i, i2w, _ = build_vocab(text, min_freq=2)
        ids = text_to_ids("咖啡森林溫度", w2i)
        assert ids[0] == w2i["咖啡"]
        assert ids[1] == UNK_ID
        assert ids[2] == UNK_ID

    def test_high_freq_kept(self):
        text = "咖啡咖啡溫度溫度森林森林"
        w2i, i2w, _ = build_vocab(text, min_freq=2)
        ids = text_to_ids("咖啡溫度森林", w2i)
        assert ids[0] == w2i["咖啡"]
        assert ids[1] == w2i["溫度"]
        assert ids[2] == w2i["森林"]

    def test_all_unknown_returns_unks(self):
        w2i, i2w, _ = build_vocab("咖啡咖啡", min_freq=2)
        ids = text_to_ids("溫度森林", w2i)
        assert all(i == UNK_ID for i in ids)

    def test_real_story_with_min_freq_3(self):
        w2i, i2w, vs = build_vocab(FALLBACK_STORY, min_freq=3)
        assert UNK_TOKEN in w2i
        assert vs == len(w2i)
        ids = text_to_ids(FALLBACK_STORY, w2i)
        assert all(0 <= i < vs for i in ids)
        assert len(ids) > 0


class TestDomainWords:
    """測試 jieba 領域詞保護功能"""

    def test_domain_words_not_split(self):
        for word in DOMAIN_WORDS:
            tokens = tokenize(word)
            assert len(tokens) == 1, (
                f"領域詞「{word}」被 jieba 切成 {tokens}，應保持為一個 token"
            )

    def test_domain_words_in_sentence_context(self):
        sentence = "精品咖啡的萃取率與水洗處理法有關"
        tokens = tokenize(sentence)
        assert "精品咖啡" in tokens, f"「精品咖啡」被 jieba 切散: {tokens}"
        assert "萃取率" in tokens, f"「萃取率」被 jieba 切散: {tokens}"
        assert "水洗" in tokens, f"「水洗」被 jieba 切散: {tokens}"

    def test_coffee_processing_terms(self):
        sentence = "淺焙和深焙的咖啡豆適合不同的沖煮方式"
        tokens = tokenize(sentence)
        assert "淺焙" in tokens, f"「淺焙」被切散: {tokens}"
        assert "深焙" in tokens, f"「深焙」被切散: {tokens}"

    def test_domain_word_appears_once_in_token_list(self):
        text = "精品咖啡精品咖啡"
        tokens = tokenize(text)
        assert tokens.count("精品咖啡") == 2, f"「精品咖啡」出現次數不正確: {tokens}"
