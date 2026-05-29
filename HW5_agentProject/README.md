# AgentSecure - AI 語音助理（含檔案存取安全控管）

一個基於 Ollama 的互動式 AI 助理，具備記憶功能與檔案存取安全機制。

## 功能特色

- **AI 對話**：使用 Ollama 的 gemma3:27b 模型
- **記憶系統**：自動提取並儲存關鍵資訊
- **命令執行**：AI 可透過 `<shell>` 標籤執行 shell 命令
- **安全控管**：檔案存取安全檢查，防止未經授權的外部檔案操作

## 安裝

### 環境需求

- Python 3.10+
- Ollama 本地服務（localhost:11434）

### 安裝依賴

```bash
pip install aiohttp
```

### 準備 Ollama 模型

```bash
ollama pull gemma3:27b
```

## 使用方法

```bash
python agentSecure.py
```

### 內部指令

| 指令 | 說明 |
|------|------|
| `/quit`, `/exit`, `/q` | 離開助理 |
| `/memory` | 顯示已儲存的关键資訊 |

## 安全機制

### 檔案存取檢查

Agent 執行命令前會檢查所有涉及的路徑：

1. **安全路徑**：位於 `~/.agentSecure` 內的檔案可直接執行
2. **外部路徑**：觸發警告並請求使用者核可（y/n）
3. **系統路徑**：`/usr/`, `/bin/`, `/sbin/`, `/lib/`, `/etc/` 等系統目錄允許執行

### 防護機制

- 防範 `..` 路徑穿越攻擊
- 防範 `~/` 展開繞過
- 自動標準化為絕對路徑檢查

### 檢查範圍

涉及檔案操作的命令會被檢查：
`cat`, `rm`, `cp`, `mv`, `ln`, `chmod`, `chown`, `touch`, `mkdir`, `rmdir`, `find`, `grep`, `head`, `tail`, `sed`, `awk`, `python3`, `python` 等。

## 使用範例

```
你：幫我建立一個測試檔案
🤖 好的，我來建立一個測試檔案

<shell>touch ~/.agentSecure/test.txt</shell>
<end/>

=== 執行命令 ===
touch ~/.agentSecure/test.txt

結果：（無輸出）

🤖 已完成！檔案已建立在 ~/.agentSecure/test.txt
```

### 外部存取攔截範例

```
🤖 讓我讀取外部檔案

<shell>cat /tmp/secret.txt</shell>

=== 執行命令 ===
cat /tmp/secret.txt

⚠️  安全警告：Agent 試圖存取外部檔案 [/tmp/secret.txt]
是否核可？(y/n): n

結果：存取遭拒絕：外部檔案存取被攔截
```

## 專案結構

```
agentProject/
├── agentSecure.py         # 主程式
├── CLAUDE.md         # 開發者指引
├── task.md           # 任務需求
└── README.md         # 本文件
```

## 核心函式說明

### `extract_paths_from_command(cmd: str) -> list[str]`
從命令字串中提取所有可能的檔案路徑。

### `is_path_safe(command: str) -> tuple[bool, list[str]]`
檢查命令中的路徑是否安全，回傳 `(是否安全，外部路徑列表)`。

### `request_external_access_approval(external_paths: list[str]) -> bool`
顯示安全警告並請求使用者核可。

### `update_memory(user_input, assistant_response, tool_result)`
更新對話歷史與關鍵資訊記憶。

## 安全注意事項

- 核可外部存取前請確認檔案來源
- 定期清理 `~/.agentSecure` 目錄
- 避免給予 AI 過度權限的提示詞

## 版本記錄

### v1.1 (2026-04-23)
- 新增檔案存取安全檢查功能
- 新增路徑穿越攻擊防護
- 新增使用者核可機制

**測試驗證結果：**
```
測試 1: cat /tmp/secret.txt
  結果：is_safe=False, external_paths=['/tmp/secret.txt']
  ✓ 被正確攔截

測試 2: cat ../outside.txt
  結果：is_safe=False, external_paths=['/home/shanghua/claudeCode/outside.txt']
  ✓ 被正確攔截

測試 3: cat ~/.agentSecure/test.txt
  結果：is_safe=True, external_paths=[]
  ✓ 被正確允許

測試 4: ls /usr/bin
  結果：is_safe=True, external_paths=[]
  ✓ 被正確允許
```

### v1.0 (2026-04-23)
- 初始版本
- AI 對話功能
- 記憶系統
