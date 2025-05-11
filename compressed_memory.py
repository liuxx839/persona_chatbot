"""
压缩记忆模块：存储和管理角色的压缩记忆
每个角色的压缩记忆是其经验的概括总结，定期从详细记忆中更新
"""

import os
import json
from datetime import datetime

# 确保记忆目录存在
MEMORY_DIR = "bot_memories"
COMPRESSED_MEMORY_FILE = os.path.join(MEMORY_DIR, "compressed_memories.json")

def ensure_memory_dir():
    """确保记忆目录存在"""
    os.makedirs(MEMORY_DIR, exist_ok=True)

def load_compressed_memories():
    """加载所有角色的压缩记忆"""
    ensure_memory_dir()
    if os.path.exists(COMPRESSED_MEMORY_FILE):
        try:
            with open(COMPRESSED_MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载压缩记忆时出错: {e}")
            return {}
    return {}

def save_compressed_memories(compressed_memories):
    """保存所有角色的压缩记忆"""
    ensure_memory_dir()
    try:
        with open(COMPRESSED_MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(compressed_memories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存压缩记忆时出错: {e}")

def update_compressed_memory(client, llm_model, persona_name, old_memory, new_memory):
    """
    使用LLM更新角色的压缩记忆
    将旧的压缩记忆与新的详细记忆合并成新的压缩版本
    """
    compressed_memories = load_compressed_memories()
    
    # 如果角色还没有压缩记忆，初始化为空
    if persona_name not in compressed_memories:
        compressed_memories[persona_name] = ""
    
    # 获取当前压缩记忆
    current_compressed = compressed_memories[persona_name]
    
    # 如果没有现有的压缩记忆，创建初始版本
    if not current_compressed:
        compressed_memories[persona_name] = f"初始压缩记忆 ({datetime.now().strftime('%Y-%m-%d')}): {new_memory[:500]}..."
        save_compressed_memories(compressed_memories)
        return compressed_memories[persona_name]
    
    # 使用LLM合并旧的压缩记忆和新的详细记忆
    try:
        prompt = (
            f"你是一个记忆压缩专家，负责为AI角色 {persona_name} 维护一个精炼的经验库。\n\n"
            f"当前压缩记忆（代表角色的核心知识和经验）:\n{current_compressed}\n\n"
            f"最近的详细记忆（需要整合进压缩记忆）:\n{new_memory}\n\n"
            f"请创建一个新的压缩记忆，它应该:\n"
            f"1. 保留核心人格特征和关键经验\n"
            f"2. 包含最重要的新发现和互动\n"
            f"3. 删除重复或琐碎的信息\n"
            f"4. 保持在1000字以内\n"
            f"5. 保留角色发展的时间线感\n\n"
            f"以精炼的第一人称形式呈现，就像这是角色自己的核心记忆。"
        )
        
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "你是一位优秀的记忆压缩专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        new_compressed = completion.choices[0].message.content.strip()
        compressed_memories[persona_name] = new_compressed
        save_compressed_memories(compressed_memories)
        return new_compressed
        
    except Exception as e:
        print(f"更新 {persona_name} 的压缩记忆时出错: {e}")
        return current_compressed

def get_compressed_memory(persona_name):
    """获取特定角色的压缩记忆"""
    compressed_memories = load_compressed_memories()
    return compressed_memories.get(persona_name, "")