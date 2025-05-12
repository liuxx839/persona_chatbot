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

def get_detailed_memory(persona_name, max_entries=20, get_all=False):
    """
    获取特定角色的详细记忆
    
    参数:
        persona_name: 角色名称
        max_entries: 默认情况下返回的最大记忆条目数
        get_all: 若为True，则返回所有记忆条目，忽略max_entries参数
    
    返回:
        格式化的记忆字符串
    """
    detailed_memories = load_detailed_memories()
    memory_entries = detailed_memories.get(persona_name, [])
    
    if not memory_entries:
        return "尚无详细记忆记录。"
    
    # 决定使用哪些记忆条目
    if get_all:
        entries_to_use = memory_entries  # 使用所有记忆条目
    else:
        # 只取最近的max_entries条记忆
        entries_to_use = memory_entries[-max_entries:] if len(memory_entries) > max_entries else memory_entries
    
    # 格式化输出选定的记忆条目
    formatted_memory = ""
    for entry in entries_to_use:
        formatted_memory += f"[{entry['timestamp']}]\n{entry['content']}\n\n"
    
    return formatted_memory
