#!/usr/bin/env python3
# agentSecure.py - AI Agent with memory and tool feedback
# Run: python agentSecure.py

import os
import asyncio
import aiohttp
import re
import shlex

# ─── Configuration ───

DEFAULT_WORKSPACE = os.path.expanduser("~/agentSecure")
DEFAULT_MODEL = "gemma3:27b"
MAX_TURNS = 5
MAX_TOOL_ITERATIONS = 5
MAX_OUTPUT_LENGTH = 2000

SYSTEM_PROMPT = """你是 Jarvc，一個有用的 AI 助理。

重要規則：
1. 當你需要執行 shell 命令時，必須用 <shell> 標籤包住命令
2. <shell> 標籤內可以是多行命令（用反斜槓 \\ 或 && 連接）
3. 你完成所有操作後，用 <end/> 結束你的回覆

流程：
- 如果需要執行命令，輸出 <shell>...</shell>
- 執行後我會顯示結果
- 如果還需要更多命令，繼續輸出 <shell>
- 當完成所有操作後，輸出 <end/> 表示結束"""

class SecureAgent:
    def __init__(self, workspace: str = DEFAULT_WORKSPACE, model: str = DEFAULT_MODEL):
        self.workspace = os.path.abspath(workspace)
        self.model = model
        self.conversation_history: list[str] = []
        self.key_info: list[str] = []
        os.makedirs(self.workspace, exist_ok=True)

    def _extract_paths_from_command(self, cmd: str) -> list[str]:
        """從命令中提取所有可能的路徑"""
        paths = []
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()

        for i, token in enumerate(tokens):
            if i == 0:
                continue

            if token.startswith('/') or token.startswith('~/'):
                paths.append(token)
            elif token.startswith('-'):
                continue
            elif '=' in token:
                key, value = token.split('=', 1)
                if value.startswith('/') or value.startswith('~/'):
                    paths.append(value)
            elif token in ('>', '>>', '<', '2>', '2>>'):
                if i + 1 < len(tokens):
                    paths.append(tokens[i + 1])
            elif token.startswith('>') or token.startswith('<'):
                paths.append(token[1:])
            else:
                if token.startswith('./') or token.startswith('../'):
                    paths.append(token)

        return paths

    def is_path_safe(self, command: str) -> tuple[bool, list[str]]:
        """
        檢查命令中的路徑是否安全。
        回傳 (is_safe, external_paths) 其中:
          - is_safe: True 表示所有路徑都在 workspace 內
          - external_paths: 外部路徑列表
        """
        workspace_normalized = os.path.normpath(self.workspace)

        def is_within_workspace(path: str) -> bool:
            abs_path = os.path.abspath(os.path.expanduser(path))
            workspace_abs = os.path.abspath(workspace_normalized)
            return abs_path.startswith(workspace_abs + os.sep) or abs_path == workspace_abs

        def is_system_command(path: str) -> bool:
            normalized = os.path.normpath(path)
            system_prefixes = ('/usr/', '/bin/', '/sbin/', '/lib/')
            return normalized.startswith(system_prefixes)

        external_paths = []
        paths = self._extract_paths_from_command(command)

        if paths:
            for path in paths:
                abs_path = os.path.abspath(os.path.expanduser(path))
                if not is_within_workspace(path) and not is_system_command(path):
                    external_paths.append(abs_path)

        return (len(external_paths) == 0, external_paths)

    async def request_external_access_approval(self, external_paths: list[str]) -> bool:
        """請求外部存取核可，回傳 True 表示核可"""
        for path in external_paths:
            print(f"\n⚠️  安全警告：Agent 試圖存取外部檔案 [{path}]")

        try:
            # Run blocking input in executor to avoid blocking the event loop
            response = await asyncio.to_thread(input, "是否核可？(y/n): ")
            return response.strip().lower() == 'y'
        except (EOFError, KeyboardInterrupt):
            print("\n存取遭拒絕")
            return False

    async def call_ollama(self, prompt: str, system: str = "") -> str:
        """Call Ollama API"""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    return result.get("response", "").strip()
            except Exception as e:
                return f"API 呼叫失敗：{e}"

    def build_context(self) -> str:
        context_parts = []
        if self.key_info:
            items_xml = "\n".join(f"  <item>{k}</item>" for k in self.key_info)
            context_parts.append(f"<memory>\n{items_xml}\n</memory>")
        if self.conversation_history:
            context_parts.append("<history>\n" + "\n".join(self.conversation_history[-MAX_TURNS*2:]) + "\n</history>")
        return "\n\n".join(context_parts)

    def update_memory(self, user_input: str, assistant_response: str, tool_result: str = None):
        self.conversation_history.append(f"  <user>{user_input}</user>")
        self.conversation_history.append(f"  <assistant>{assistant_response}</assistant>")
        if tool_result:
            self.conversation_history.append(f"  <tool>{tool_result[:500]}</tool>")
        
        while len(self.conversation_history) > MAX_TURNS * 4:
            self.conversation_history.pop(0)

    async def extract_key_info(self, user_input: str, assistant_response: str):
        extract_prompt = f"""根據這段對話，有沒有需要長期記憶的關鍵資訊？
如果有，用以下格式輸出（最多 2 項）。如果沒有，輸出 <memory></memory>。

<memory>
  <item>要記憶的資訊 1</item>
  <item>要記憶的資訊 2</item>
</memory>

對話：
<user>{user_input}</user>
<assistant>{assistant_response}</assistant>"""
        
        try:
            result = await self.call_ollama(extract_prompt, "")
            matches = re.findall(r'<item>(.*?)</item>', result, re.DOTALL)
            for item in matches:
                item = item.strip()
                if item and item not in self.key_info:
                    self.key_info.append(item)
        except Exception as e:
            # Don't fail the whole turn just because memory extraction failed
            print(f"[Debug] Memory extraction failed: {e}")

    async def run_shell_command(self, cmd: str) -> str:
        """非同步執行 shell 命令並截斷過長的輸出"""
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                process.kill()
                return "錯誤：命令執行逾時 (超過 30 秒)"
                
            output = ""
            if stdout:
                output += stdout.decode(errors='replace')
            if stderr:
                output += stderr.decode(errors='replace')
                
            if not output:
                return "（無輸出）"
                
            if len(output) > MAX_OUTPUT_LENGTH:
                return output[:MAX_OUTPUT_LENGTH] + f"\n\n... (輸出過長，已截斷，隱藏了 {len(output) - MAX_OUTPUT_LENGTH} 字元) ..."
            return output
        except Exception as e:
            return f"執行命令時發生錯誤：{e}"

    async def chat_loop(self):
        print(f"agentSecure - {self.model}（含記憶功能）")
        print(f"工作區：{self.workspace}")
        print("指令：/quit、/memory（顯示關鍵資訊）\n")
        
        while True:
            try:
                user_input = await asyncio.to_thread(input, "你：")
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再見！")
                break
            
            if not user_input:
                continue
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("再見！")
                break
            if user_input.lower() == "/memory":
                print(f"關鍵資訊：{self.key_info}")
                continue
            
            context = self.build_context()
            full_prompt = f"{context}\n\n<user>{user_input}</user>" if context else f"<user>{user_input}</user>"
            
            response = await self.call_ollama(full_prompt, SYSTEM_PROMPT)
            
            tool_result = None
            current_response = response
            final_response_parts = []
            
            iterations = 0

            while iterations < MAX_TOOL_ITERATIONS:
                iterations += 1
                
                response_content = current_response.split("<end/>")[0].strip()
                shell_matches = re.findall(r'<shell>(.+?)</shell>', current_response, re.DOTALL)
                
                if not shell_matches:
                    if response_content:
                        final_response_parts.append(response_content)
                    break

                if response_content:
                    final_response_parts.append(response_content)

                command_outputs = []
                for cmd in shell_matches:
                    cmd = cmd.strip()

                    is_safe, external_paths = self.is_path_safe(cmd)

                    if not is_safe:
                        if not await self.request_external_access_approval(external_paths):
                            output_text = "存取遭拒絕：外部檔案存取被攔截"
                            command_outputs.append(f"$ {cmd}\n{output_text}")
                            print(f"\n⚠️  {output_text}\n")
                            continue

                    print(f"\n=== 執行命令 ===\n{cmd}")
                    output_text = await self.run_shell_command(cmd)
                    print(f"結果：\n{output_text}\n")
                    command_outputs.append(f"$ {cmd}\n{output_text}")

                tool_result = (tool_result or "") + "\n" + "\n".join(command_outputs)

                if command_outputs:
                    # Provide a preview to the user without overwhelming the terminal if output is long
                    preview = "\n".join(command_outputs)
                    if len(preview) > 500:
                         preview = preview[:500] + "\n... [Output Truncated for Console] ..."
                    print(f"🤖 {preview}\n")

                follow_up_prompt = f"""<context>{context}</context>

<user>{user_input}</user>
<assistant>{current_response}</assistant>
<output>
{chr(10).join(command_outputs)}
</output>

如果需要更多命令就輸出 <shell>。否則，輸出 <end/> 表示結束："""
                
                current_response = await self.call_ollama(follow_up_prompt, SYSTEM_PROMPT)
            
            if iterations >= MAX_TOOL_ITERATIONS:
                print("⚠️  達到最大工具呼叫次數限制，終止本回合。")
                final_response_parts.append("\n[系統提示] 達到最大內部迴圈限制，終止操作。")

            if final_response_parts:
                response = "\n\n".join(final_response_parts)
            else:
                response = current_response.split("<end/>")[0].strip()
            
            print(f"\n🤖 {response}\n")
            
            self.update_memory(user_input, response, tool_result)
            if tool_result:
                # Fire and forget memory extraction or await it.
                # Since we want to update memory before next turn, we await it but it's fast.
                await self.extract_key_info(user_input, response)

async def main():
    agent = SecureAgent()
    await agent.chat_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程式已終止。")
