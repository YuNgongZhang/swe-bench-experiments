# 完整的代码提示压缩工具
import os
import re
import tiktoken
from pathlib import Path
from openai import OpenAI
from rapidfuzz import fuzz  # 用于文本相似度匹配
from collections import defaultdict

class CodePromptCompressor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        return len(self.encoding.encode(text))
    
    def compress_instructions(self, text, preserve_format=True):
        """
        对非代码文本进行压缩。保留原有 placeholders。
        注意：对于问题描述，只压缩而不删除。
        """
        system_prompt = """You are an expert at compressing programming task instructions while preserving all essential details.
        
        Your job is to:
        1. Remove redundant explanations and verbose text, but KEEP all problem statements
        2. Preserve ALL technical requirements, constraints, and expected behaviors
        3. Maintain the original meaning but express it more concisely
        4. Keep all code examples EXACTLY as they are
        5. Preserve any placeholders like {{variable}} or {placeholder}
        6. Maintain all line breaks that separate distinct points
        7. Keep ALL function names, parameter details, and return value descriptions
        8. IMPORTANT: For any text that starts with "Problem:", ensure that the problem statement is preserved
        
        The output should be functionally identical but use fewer tokens. Never remove any problem statements entirely.
        """
        # 调用 GPT-3.5-turbo 接口
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Compress these instructions while preserving all problem statements and technical requirements:\n\n{text}"}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def compress_code_prompt(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_tokens = self.count_tokens(content)
        
        # =============== 第1步：拆分成 segments（code / text）===============
        # 更新：匹配代码片段和补丁
        code_pattern = r'(## Code Snippet \d+\s*\n\s*```[\s\S]*?```|Patch: diff --git[\s\S]*?(?=\n\n|$))'
        code_matches = list(re.finditer(code_pattern, content))
        
        segments = []
        last_end = 0
        
        for match in code_matches:
            # 非代码
            if match.start() > last_end:
                non_code = content[last_end:match.start()]
                if non_code.strip():
                    segments.append({"type": "text", "content": non_code})
            # 代码或补丁
            segments.append({"type": "code", "content": match.group(0)})
            last_end = match.end()
        
        # 末尾剩余非代码
        if last_end < len(content):
            non_code = content[last_end:]
            if non_code.strip():
                segments.append({"type": "text", "content": non_code})
        
        # =============== 第2步：从 segments 中找出 problem 段落，做相似度分析 ================
        # 先提取所有 "problem" 文本
        problem_texts = []
        for i, seg in enumerate(segments):
            if seg["type"] == "text":
                # 用正则或简单方式匹配 "Problem:" 之后的一段文本
                problem_match = re.search(r'(Problem:\s*)(.*)', seg["content"], flags=re.IGNORECASE|re.DOTALL)
                if problem_match:
                    # entire match
                    p_text = problem_match.group(2).strip()
                    problem_texts.append({"index": i, "text": p_text})
        
        # 做相似度聚类：简单做 pairwise，若相似度>80，就认定为同一组
        # 修复：直接比较文本内容，不使用索引
        groups = []  # list of lists
        threshold = 80  # 相似度阈值
        for i, pb in enumerate(problem_texts):
            found_group = False
            for g in groups:
                # 拿 group 里第一个元素来比较
                ref_pb = g[0]
                score = fuzz.ratio(pb["text"], ref_pb["text"])
                if score >= threshold:
                    g.append(pb)
                    found_group = True
                    break
            if not found_group:
                new_group = [pb]
                groups.append(new_group)
        
        # 构造映射： segment_index -> group_id
        seg_index_to_group = {}
        for g_id, g_members in enumerate(groups):
            for mem in g_members:
                seg_index_to_group[mem["index"]] = g_id
        
        # =============== 第3步：合并同组 problem 段落 ================
        # 思路：新建 segments2，把同一组 problem 的文本合并成一块
        # 其他非-problem text 或 code 保持原样
        merged_problems = {}  # group_id -> merged text
        for g_id in range(len(groups)):
            merged_problems[g_id] = ""  # 初始化
        
        new_segments = []
        visited = set()
        
        for i, seg in enumerate(segments):
            if seg["type"] == "text":
                if i in seg_index_to_group:
                    g_id = seg_index_to_group[i]
                    if g_id not in visited:
                        # 第一次遇到该组 -> 合并所有组内 problem
                        # 先找组内所有 index
                        lines_to_combine = []
                        for mem in groups[g_id]:
                            # 取出原 segments[mem["index"]]["content"]
                            lines_to_combine.append(segments[mem["index"]]["content"])
                        
                        combined_text = "\n\n".join(lines_to_combine)
                        # 压缩后再添加到 new_segments
                        compressed_text = self.compress_instructions(combined_text)
                        new_segments.append({"type": "text", "content": compressed_text})
                        visited.add(g_id)
                    else:
                        # 如果同组 problem 已处理过，就跳过
                        continue
                else:
                    # 普通非-problem文本，正常压缩
                    compressed_text = self.compress_instructions(seg["content"])
                    new_segments.append({"type": "text", "content": compressed_text})
            
            elif seg["type"] == "code":
                # 代码段和补丁完全保持原样
                new_segments.append(seg)
        
        # =============== 第4步：组装输出并计算压缩度 ================
        final_content = ""
        for seg in new_segments:
            final_content += seg["content"]
        
        compressed_tokens = self.count_tokens(final_content)
        reduction = 1 - (compressed_tokens / original_tokens)
        
        # 写文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"Original tokens: {original_tokens}")
        print(f"Compressed tokens: {compressed_tokens}")
        print(f"Reduction: {reduction:.2%}")
        print(f"Compressed file saved to: {output_file}")
        
        return {
            "original_file": input_file,
            "compressed_file": output_file,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "reduction": reduction
        }

# 示例用法
if __name__ == "__main__":
    compressor = CodePromptCompressor()
    result = compressor.compress_code_prompt("all_issues_and_patches.md", "compressed_code.md")
    print(f"Token reduction: {result['reduction']:.2%}")
