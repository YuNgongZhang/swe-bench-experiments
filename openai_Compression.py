import os
import tiktoken
import re
from pathlib import Path
from openai import OpenAI

class CodePromptCompressor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        return len(self.encoding.encode(text))
    
    def extract_and_replace_code_blocks(self, text):
        """Extract code blocks and replace with placeholders, returning the list of code blocks"""
        pattern = r'(```[\w]*\n[\s\S]*?```)'
        blocks = []
        
        # Find all code blocks
        matches = re.finditer(pattern, text)
        replaced_text = text
        
        for i, match in enumerate(matches):
            placeholder = f"[CODE_BLOCK_{i}]"
            blocks.append(match.group(1))  # Save the entire code block (including ``` markers)
            replaced_text = replaced_text.replace(match.group(1), placeholder, 1)
            
        return replaced_text, blocks
    
    def restore_code_blocks(self, text, code_blocks):
        """Restore replaced code blocks"""
        result = text
        for i, block in enumerate(code_blocks):
            placeholder = f"[CODE_BLOCK_{i}]"
            result = result.replace(placeholder, block, 1)
        return result
    
    def extract_and_replace_patch_blocks(self, text):
        """Specifically extract and replace <patch>...</patch> blocks"""
        pattern = r'(<patch>[\s\S]*?<\/patch>)'
        blocks = []
        
        # Find all patch blocks
        matches = re.finditer(pattern, text)
        replaced_text = text
        
        for i, match in enumerate(matches):
            placeholder = f"[PATCH_BLOCK_{i}]"
            blocks.append(match.group(1))  # Save the entire patch block
            replaced_text = replaced_text.replace(match.group(1), placeholder, 1)
            
        return replaced_text, blocks
    
    def restore_patch_blocks(self, text, patch_blocks):
        """Restore replaced patch blocks"""
        result = text
        for i, block in enumerate(patch_blocks):
            placeholder = f"[PATCH_BLOCK_{i}]"
            result = result.replace(placeholder, block, 1)
        return result
    
    def compress_instructions(self, text, preserve_format=True):
        """Compress the instruction part, maintaining key directives"""
        
        system_prompt = """You are an expert at compressing programming task instructions. 
Your goal is to make instructions as concise as possible while preserving all necessary information.
IMPORTANT: Preserve all placeholders exactly as they appear (like [CODE_BLOCK_0] or [PATCH_BLOCK_0]).
DO NOT rewrite, modify or remove these placeholders!"""
        
        if preserve_format:
            system_prompt += "\nPreserve headings (## Code Snippet X) and other formatting markers."
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Compress these instructions to absolute minimum necessary words while preserving critical information and ALL placeholders:\n\n{text}"}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
        
    def compress_code_prompt(self, input_file, output_file):
        """Compress code fix prompts, preserving code blocks and patch blocks integrity"""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Calculate original tokens
        original_tokens = self.count_tokens(content)
        
        # Use segmented processing to identify all code and non-code segments
        # Pattern to recognize code segments
        code_pattern = r'## Code Snippet \d+\s*\n\s*```[\s\S]*?```'
        
        # Find all code blocks (including headings)
        code_matches = list(re.finditer(code_pattern, content))
        
        # Create segment list, alternating between non-code and code segments
        segments = []
        last_end = 0
        
        for match in code_matches:
            # Add text before code block
            if match.start() > last_end:
                non_code = content[last_end:match.start()]
                if non_code.strip():
                    segments.append({"type": "text", "content": non_code})
            
            # Add code block
            segments.append({"type": "code", "content": match.group(0)})
            last_end = match.end()
        
        # Add the last non-code segment (if it exists)
        if last_end < len(content):
            non_code = content[last_end:]
            if non_code.strip():
                segments.append({"type": "text", "content": non_code})
        
        # Only compress non-code segments
        for i, segment in enumerate(segments):
            if segment["type"] == "text":
                # Only compress non-code segments
                segments[i]["content"] = self.compress_instructions(segment["content"])
        
        # Recombine content
        final_content = ""
        for segment in segments:
            final_content += segment["content"]
        
        # Calculate compressed tokens
        compressed_tokens = self.count_tokens(final_content)
        reduction = 1 - (compressed_tokens / original_tokens)
        
        # Write to output file
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

cp = CodePromptCompressor()
result = cp.compress_code_prompt("extracted_code.md", "compressed_code.md")