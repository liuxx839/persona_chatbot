"""
详细记忆模块：存储和管理角色的完整记忆历史
保存所有原始bot_memory内容，不进行任何压缩或删减
"""

import os
import json
from datetime import datetime

# 确保记忆目录存在
MEMORY_DIR = "bot_memories"
DETAILED_MEMORY_FILE = os.path.join(MEMORY_DIR, "detailed_memories.json")

def ensure_memory_dir():
    """确保记忆目录存在"""
    os.makedirs(MEMORY_DIR, exist_ok=True)

def load_detailed_memories():
    """加载所有角色的详细记忆"""
    ensure_memory_dir()
    if os.path.exists(DETAILED_MEMORY_FILE):
        try:
            with open(DETAILED_MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载详细记忆时出错: {e}")
            return {}
    return {}

def save_detailed_memories(detailed_memories):
    """保存所有角色的详细记忆"""
    ensure_memory_dir()
    try:
        with open(DETAILED_MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(detailed_memories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存详细记忆时出错: {e}")

def append_to_detailed_memory(persona_name, memory_entry):
    """向角色的详细记忆添加新条目"""
    detailed_memories = load_detailed_memories()
    
    # 如果角色还没有详细记忆，初始化为空列表
    if persona_name not in detailed_memories:
        detailed_memories[persona_name] = []
    
    # 添加带时间戳的新记忆
    memory_with_timestamp = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": memory_entry
    }
    
    detailed_memories[persona_name].append(memory_with_timestamp)
    save_detailed_memories(detailed_memories)

def get_detailed_memory(persona_name):
    """获取特定角色的完整详细记忆"""
    detailed_memories = load_detailed_memories()
    memory_entries = detailed_memories.get(persona_name, [])
    
    if not memory_entries:
        return "尚无详细记忆记录。"
    
    # 格式化输出所有记忆条目
    formatted_memory = ""
    for entry in memory_entries:
        formatted_memory += f"[{entry['timestamp']}]\n{entry['content']}\n\n"
    
    return formatted_memory