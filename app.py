import streamlit as st
from zhipuai import ZhipuAI
from openai import OpenAI
import random
import time
from datetime import datetime
from persona import PERSONAS # Import personas
from groq import Groq
import compressed_memory as cm
import detailed_memory as dm
import os

# --- 配置 ---
MAX_HISTORY_LEN = 20 # LLM 可见的最大消息数
MAX_BOT_MEMORY_LEN = 20 # 每个机器人工作记忆的最大条目数
SUMMARY_INTERVAL = 3 # 每隔多少轮总消息进行一次总结
BOT_RESPONSE_DELAY = (0, 0) # 机器人响应延迟秒数范围（最小值，最大值）
MEMORY_COMPRESSION_INTERVAL = 3 # 每隔多少次记忆更新压缩一次记忆

# --- Groq API 设置 ---
try:
    # client = ZhipuAI(api_key='')
    # LLM_MODEL = "glm-4-flash" # Zhipu AI model
    # client = Groq(api_key='')
    # LLM_MODEL = "llama3-70b-8192" # Groq 模型
    api_key = os.environ.get("GEMINI_API_KEY")
    client = OpenAI(api_key=api_key,base_url="https://generativelanguage.googleapis.com/v1beta/")
    LLM_MODEL = "gemini-2.0-flash" # Groq 模型

except KeyError:
    st.error("API 密钥未找到。请在 .streamlit/secrets.toml 中设置它")
    st.stop()


# --- 辅助函数 ---

def get_llm_response(persona_name, persona_details, chat_history, bot_memory, compressed_memory_text=""):
    """
    根据角色、历史记录和记忆获取 LLM 的响应。
    增加了压缩记忆参数，为角色提供长期经验。
    """
    system_prompt = (
        f"你是 {persona_name}。{persona_details['description']}\n"
        f"你的背景：{persona_details['background']}\n"
    )
    
    # 添加压缩记忆（如果有）
    if compressed_memory_text:
        system_prompt += f"你的核心经验（长期记忆）：{compressed_memory_text}\n\n"
    
    system_prompt += (
        f"你的最近工作记忆（用于保持一致性）：{bot_memory}\n\n"
        f"你正在一个聊天室中。以下是最近的对话历史（最后 {MAX_HISTORY_LEN} 条消息）。"
        f"自然地进行互动。简明扼要。保持角色特性。除非这是你的第一条消息，否则不要问候。"
    )

    messages_for_llm = [{"role": "system", "content": system_prompt}]
    
    # 修复：将所有角色映射为 API 的 "user" 或 "assistant"
    for msg in chat_history[-MAX_HISTORY_LEN:]:
        # 如果消息来自当前角色，则是 "assistant" 消息
        # 否则，是 "user" 消息（无论是来自用户还是其他机器人）
        if msg["role"] == persona_name:
            role = "assistant"
            msg_content = msg["content"]
        else:
            role = "user"
            # 为了上下文，在内容中添加实际发言者的名称
            msg_content = f"[{msg['role']}]: {msg['content']}"
        
        messages_for_llm.append({
            "role": role, 
            "content": msg_content
        })

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages_for_llm,
            temperature=0.7,
            max_tokens=150
        )
        response = completion.choices[0].message.content.strip()
        return response
    except Exception as e:
        st.error(f"{persona_name} 与 LLM 通信出错：{str(e)}")
        return f"({persona_name} 思考遇到了困难...)"


def update_bot_memory(persona_name, persona_details, chat_history, current_memory):
    """
    使用 LLM 更新机器人的记忆。
    保持工作记忆在合理长度内（MAX_BOT_MEMORY_LEN条目）。
    """
    if not chat_history:
        return current_memory # 没有新信息

    # 提取与机器人相关的最近对话片段
    relevant_history_snippet = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-MAX_HISTORY_LEN:]])

    prompt = (
        f"你是一个AI助手，帮助 {persona_name}（{persona_details['description']}）更新其记忆。\n"
        f"当前记忆：\n{current_memory}\n\n"
        f"最近对话片段：\n{relevant_history_snippet}\n\n"
        f"基于此，为 {persona_name} 提供一个简洁的更新记忆。"
        f"关注 {persona_name} 应该记住的关键新事实、决定或表达/观察到的强烈感受。"
        f"保持简短，像个人笔记。如果没有重要的内容可添加，可以说'没有重要更新'。"
    )
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个有帮助的记忆助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        update = completion.choices[0].message.content.strip()
        if "没有重要更新" in update.lower() or "no significant updates" in update.lower():
            return current_memory
            
        # 添加新记忆
        new_memory_entry = f"- (更新于 {datetime.now().strftime('%H:%M')}) {update}"
        
        # 将更新添加到详细记忆（永久存储）
        dm.append_to_detailed_memory(persona_name, new_memory_entry)
        
        # 更新会话工作记忆（保持在MAX_BOT_MEMORY_LEN条以内）
        memory_lines = current_memory.split('\n')
        # 保留初始记忆行和最近的MAX_BOT_MEMORY_LEN-1条记忆
        initial_memory = memory_lines[0] if memory_lines else ""
        recent_memories = memory_lines[1:] if len(memory_lines) > 1 else []
        
        if len(recent_memories) >= MAX_BOT_MEMORY_LEN - 1:
            recent_memories = recent_memories[-(MAX_BOT_MEMORY_LEN - 2):]
        
        updated_memory = initial_memory + "\n" + "\n".join(recent_memories + [new_memory_entry])
        
        # 检查是否需要更新压缩记忆
        if "memory_updates_count" not in st.session_state:
            st.session_state.memory_updates_count = {}
        
        if persona_name not in st.session_state.memory_updates_count:
            st.session_state.memory_updates_count[persona_name] = 0
        
        st.session_state.memory_updates_count[persona_name] += 1
        
        # 定期将详细记忆压缩到压缩记忆中
        if st.session_state.memory_updates_count[persona_name] % MEMORY_COMPRESSION_INTERVAL == 0:
            # 获取完整的详细记忆用于压缩
            full_detailed_memory = dm.get_detailed_memory(persona_name,max_entries=MAX_BOT_MEMORY_LEN)
            # 获取当前压缩记忆
            current_compressed = cm.get_compressed_memory(persona_name)
            # 更新压缩记忆
            cm.update_compressed_memory(client, LLM_MODEL, persona_name, current_compressed, full_detailed_memory)
            st.toast(f"{persona_name} 的长期记忆已更新！")
        
        return updated_memory
    except Exception as e:
        st.error(f"更新 {persona_name} 的记忆时出错：{e}")
        return current_memory


def get_conversation_summary(chat_history_to_summarize):
    """
    总结一段对话。
    """
    if not chat_history_to_summarize:
        return "没有对话可总结。"

    prompt = (
        "总结以下聊天对话。突出关键话题、决定、任何冲突或协议，以及讨论的整体进展。要简明扼要。\n\n"
        "对话：\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_to_summarize])
    )
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一位总结专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        summary = completion.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"生成总结时出错：{e}")
        return "由于错误，无法生成总结。"


def determine_next_speaker(history, available_bots, last_speaker, user_name):
    """
    使用LLM作为隐藏导演，根据聊天历史分析，决定下一个发言的机器人。
    """
    if not available_bots:
        return None
    
    # 避免同一发言者连续发言的基本规则
    eligible_bots = [b for b in available_bots if b != last_speaker]
    if not eligible_bots:
        eligible_bots = available_bots
    
    # 基本情况：如果历史太短，直接随机选择
    if len(history) < 2:
        return random.choice(eligible_bots)
    
    # 获取最近的消息上下文（受MAX_HISTORY_LEN限制）
    recent_messages = history[-min(MAX_HISTORY_LEN, len(history)):]
    recent_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    
    # 快速检查：如果明确提及某个机器人，让它来回答
    last_message = recent_messages[-1]
    last_content = last_message["content"].lower()
    for bot_name in available_bots:
        if bot_name.lower() in last_content and "?" in last_content:
            return bot_name
    
    # 给一定概率随机回复，保持对话活跃性
    if random.random() < 0.15:  # 15%的概率随机选择
        return random.choice(eligible_bots)
    
    # 使用LLM来决定谁是最合适的下一个发言者
    try:
        # 准备可用角色的简短描述，帮助LLM做决定
        bot_descriptions = []
        for bot_name in eligible_bots:
            if bot_name in st.session_state.bot_personas_data:
                short_desc = st.session_state.bot_personas_data[bot_name].get('description', '')
                # 仅取简短描述的前30个字符
                bot_descriptions.append(f"- {bot_name}: {short_desc[:50]}...")
        
        bot_info = "\n".join(bot_descriptions)
        
        prompt = (
            f"你是一个聊天室的隐形导演。基于最近的对话历史，请决定谁应该是下一个发言者。\n\n"
            f"当前聊天室成员：\n{bot_info}\n\n"
            f"最近的对话历史：\n{recent_history_text}\n\n"
            f"上一个发言者是：{last_speaker}\n\n"
            f"请仔细分析对话内容，考虑以下因素：\n"
            f"1. 谁是对话中被提及或被询问的对象\n"
            f"2. 谁最适合对最近的话题进行回应（基于角色背景）\n"
            f"3. 谁可以提供有价值的新观点\n"
            f"4. 谁在最近几轮对话中发言较少\n\n"
            f"根据以上分析，从以下列表中选择一个名字作为下一个发言者（只返回名字，不要任何解释）：{', '.join(eligible_bots)}"
        )
        
        # 调用LLM获取推荐的下一个发言者
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一位对话管理专家，帮助决定谁应该是对话中的下一个发言者。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        next_speaker_suggestion = completion.choices[0].message.content.strip()
        
        # 清理响应，确保我们得到的是一个有效的发言者名称
        for bot_name in eligible_bots:
            if bot_name.lower() in next_speaker_suggestion.lower():
                return bot_name
                
        # 如果LLM没有返回有效的机器人名称，使用备选方案
        return random.choice(eligible_bots)
        
    except Exception as e:
        st.error(f"使用LLM决定下一个发言者时出错：{e}")
        # 出错时回退到随机选择
        return random.choice(eligible_bots)
    
def get_avatar_url(persona_name):
    """
    Retrieve the avatar URL for a persona from bot_personas_data.
    """
    persona = st.session_state.bot_personas_data.get(persona_name, {})
    return persona.get("avatar", f"https://api.dicebear.com/9.x/personas/svg?seed={persona_name}")

# --- Streamlit 应用 ---

st.set_page_config(page_title="LLM 聊天室", layout="wide")
st.title("🤖💬 LLM 角色聊天室")

# --- 初始化会话状态 ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # 完整聊天历史：{"role": "name", "content": "text", "timestamp": datetime}
if "bot_personas_data" not in st.session_state:
    st.session_state.bot_personas_data = {p["name"]: p for p in PERSONAS}
    # Add avatar to each persona if not present
    for name, details in st.session_state.bot_personas_data.items():
        if "avatar" not in details:
            # Generate avatar using DiceBear with persona name as seed
            details["avatar"] = f"https://api.dicebear.com/9.x/personas/svg?seed={name}"
if "bot_memories" not in st.session_state:
    st.session_state.bot_memories = {}
    # 初始化每个角色的工作记忆，并从压缩记忆加载
    for name, details in st.session_state.bot_personas_data.items():
        # 获取压缩记忆（如果有）
        compressed_memory_text = cm.get_compressed_memory(name)
        # 创建初始工作记忆
        st.session_state.bot_memories[name] = f"初始记忆：我的名字是 {name}。{details.get('background', '')}"

if "conversation_rounds" not in st.session_state:
    st.session_state.conversation_rounds = 0 # 已发送的总消息数
if "summaries" not in st.session_state:
    st.session_state.summaries = [] # 总结列表
if "last_speaker" not in st.session_state:
    st.session_state.last_speaker = None # 避免立即自我回复
if "user_name" not in st.session_state:
    st.session_state.user_name = f"用户_{random.randint(1000, 9999)}"
if "bots_in_chat" not in st.session_state:
    st.session_state.bots_in_chat = [p["name"] for p in PERSONAS[:16]] # 以前3个机器人开始
if "memory_updates_count" not in st.session_state:
    st.session_state.memory_updates_count = {} # 记录每个角色记忆更新的次数

# --- 侧边栏控件 ---
with st.sidebar:
    st.header("聊天室控制")
    st.session_state.user_name = st.text_input("你的名字", value=st.session_state.user_name)

    # 新增：用户上传persona内容
    st.subheader("添加自定义角色")
    uploaded_persona = st.text_area("输入你的自定义角色描述（例如背景、职责等）", height=150)
    if st.button("添加自定义角色"):
        if uploaded_persona:
            try:
                # 使用LLM将用户输入转换为persona格式
                prompt = (
                    f"你是一个助手，将用户提供的角色描述转换为以下JSON格式的persona结构：\n"
                    f"{{\n"
                    f'  "name": "姓名",\n'
                    f'  "description": "角色职责和特点描述",\n'
                    f'  "background": "角色的教育、经验或专业背景",\n'
                    f'  "greeting": "角色在聊天室中的开场白"\n'
                    f"}}\n\n"
                    f"用户提供的描述：\n{uploaded_persona}\n\n"
                    f"请分析输入，提取关键信息，生成一个符合上述格式的persona。确保内容简洁、符合角色设定，且greeting自然且与角色背景相关。"
                    f"直接返回符合JSON格式的persona对象，不要有任何额外说明或解释。"
                    f"如果用户输入不包含足够的角色信息，请拒绝生成并返回字符串'INVALID_INPUT'。"
                )
                completion = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个擅长提取和格式化信息的助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=300
                )
                result = completion.choices[0].message.content.strip()
                # print(result)
                # 检查是否为无效输入
                if result == "INVALID_INPUT":
                    st.warning("输入内容不足以创建有意义的角色。请提供更详细的角色描述。")
                else:
                    # 解析LLM返回的JSON
                    import json
                    # 查找JSON内容的开始和结束位置
                    json_start = result.find("{")
                    json_end = result.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        new_persona_json = result[json_start:json_end]
                        new_persona = json.loads(new_persona_json)
                        
                        # 验证必要字段
                        required_fields = ["name", "description", "background", "greeting"]
                        if all(field in new_persona for field in required_fields):
                            new_persona["avatar"] = f"https://api.dicebear.com/9.x/personas/svg?seed={new_persona['name']}"
                            # 直接添加到PERSONAS列表
                            PERSONAS.append(new_persona)
                            
                            # 同时更新会话状态
                            st.session_state.bot_personas_data[new_persona["name"]] = new_persona
                            st.session_state.bots_in_chat.append(new_persona["name"])
                            st.session_state.bot_memories[new_persona["name"]] = (
                                f"初始记忆：我的名字是 {new_persona['name']}。{new_persona['background']}"
                            )
                            st.success(f"已成功添加角色 {new_persona['name']} 到模拟对话角色列表！")
                        else:
                            st.error("生成的persona缺少必要字段，请提供更详细的角色描述。")
                    else:
                        st.error("无法解析返回的JSON格式，请检查输入内容或尝试重新提交。")
                        
            except Exception as e:
                st.error(f"处理自定义角色时出错：{str(e)}")
                st.code(result, language="json")  # 显示原始返回结果以便调试
        else:
            st.warning("请提供角色描述后再添加。")

    available_bots = [p["name"] for p in PERSONAS]
    st.session_state.bots_in_chat = st.multiselect(
        "选择聊天中的机器人",
        options=available_bots,
        default=st.session_state.bots_in_chat
    )

    if not st.session_state.bots_in_chat:
        st.warning("请至少选择一个机器人进行聊天。")

    st.subheader("机器人个性与记忆")
    for bot_name in st.session_state.bots_in_chat:
        persona = st.session_state.bot_personas_data.get(bot_name)
        if persona:
            with st.expander(f"{bot_name} 的角色与记忆"):
                st.markdown(f"**描述:** {persona['description']}")
                st.markdown(f"**背景:** {persona['background']}")
                
                # 显示工作记忆（会话中显示的记忆）
                st.markdown("**工作记忆:**")
                st.text_area(f"{bot_name} 的工作记忆", 
                             value=st.session_state.bot_memories.get(bot_name, "尚无记忆。"), 
                             height=100, 
                             key=f"mem_{bot_name}_display", 
                             disabled=True)
                
                # 添加压缩记忆显示
                compressed_mem = cm.get_compressed_memory(bot_name)
                st.markdown("**压缩长期记忆:**")
                st.text_area(f"{bot_name} 的压缩记忆", 
                             value=compressed_mem if compressed_mem else "尚无压缩记忆。", 
                             height=100, 
                             key=f"compressed_{bot_name}_display", 
                             disabled=True)
                
                # 添加详细记忆查看按钮
                if st.button(f"查看 {bot_name} 的完整详细记忆", key=f"view_detailed_{bot_name}"):
                    detailed_mem = dm.get_detailed_memory(bot_name,get_all=True)
                    st.session_state[f"show_detailed_{bot_name}"] = True
                    st.session_state[f"detailed_mem_{bot_name}"] = detailed_mem
                
                # 显示详细记忆（如果按钮被点击）
                if st.session_state.get(f"show_detailed_{bot_name}", False):
                    st.markdown("**详细记忆历史:**")
                    st.text_area(f"{bot_name} 的详细记忆历史", 
                                value=st.session_state.get(f"detailed_mem_{bot_name}", "尚无详细记忆。"), 
                                height=300, 
                                key=f"detailed_{bot_name}_display", 
                                disabled=True)
                    if st.button("关闭详细记忆", key=f"close_detailed_{bot_name}"):
                        st.session_state[f"show_detailed_{bot_name}"] = False

    st.subheader("对话总结")
    if st.session_state.summaries:
        for i, summary_text in enumerate(st.session_state.summaries):
            with st.expander(f"总结 {i+1} (第 { (i+1) * SUMMARY_INTERVAL } 轮后)"):
                st.markdown(summary_text)
    else:
        st.caption("尚未生成总结。")

    if st.button("强制机器人回复"):
        st.session_state.force_bot_turn = True
        st.rerun() # 重新运行以触发机器人回合逻辑

    # 新增：机器人自动按钮和轮数输入
    st.subheader("自动对话")
    auto_rounds = st.number_input("指定自动对话轮数", min_value=1, max_value=50, value=5, step=1)
    if st.button("机器人自动"):
        st.session_state.auto_bot_turns = auto_rounds
        st.session_state.current_auto_turn = 0
        st.rerun()


# --- 显示聊天消息 ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == st.session_state.user_name else get_avatar_url(msg["role"])
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(f"**{msg['role']}** ({msg['timestamp'].strftime('%H:%M:%S')}):")
            st.write(msg["content"])


# --- 处理机器人回合（自主聊天）---
def bot_autonomous_turn():
    if not st.session_state.bots_in_chat:
        return False

    # 使用隐藏导演确定下一个发言者
    chosen_bot_name = determine_next_speaker(
        st.session_state.messages, 
        st.session_state.bots_in_chat,
        st.session_state.last_speaker,
        st.session_state.user_name
    )
    
    if chosen_bot_name:
        persona_details = st.session_state.bot_personas_data[chosen_bot_name]
        bot_memory = st.session_state.bot_memories[chosen_bot_name]
        #获取压缩记忆
        compressed_memory_text = cm.get_compressed_memory(chosen_bot_name)
        # time.sleep(random.uniform(BOT_RESPONSE_DELAY[0], BOT_RESPONSE_DELAY[1]))

        with st.spinner(f"{chosen_bot_name} 正在输入..."):
            bot_response = get_llm_response(
                chosen_bot_name,
                persona_details,
                st.session_state.messages,
                bot_memory,
                compressed_memory_text
            )

        if bot_response:
            timestamp = datetime.now()
            st.session_state.messages.append({"role": chosen_bot_name, "content": bot_response, "timestamp": timestamp})
            st.session_state.last_speaker = chosen_bot_name
            st.session_state.conversation_rounds += 1

            new_memory = update_bot_memory(chosen_bot_name, persona_details, st.session_state.messages, bot_memory)
            st.session_state.bot_memories[chosen_bot_name] = new_memory
            return True
    return False

# --- 处理强制机器人回合或决定机器人是否应该说话 ---
if st.session_state.get("force_bot_turn", False):
    st.session_state.force_bot_turn = False
    if bot_autonomous_turn():
        st.rerun()

# --- 用户输入 ---
if prompt := st.chat_input(f"以 {st.session_state.user_name} 的身份聊天..."):
    timestamp = datetime.now()
    st.session_state.messages.append({"role": st.session_state.user_name, "content": prompt, "timestamp": timestamp})
    st.session_state.last_speaker = st.session_state.user_name
    st.session_state.conversation_rounds += 1

    bot_responded_after_user = bot_autonomous_turn()
    st.rerun()

# --- 总结逻辑 ---
if st.session_state.conversation_rounds > 0 and \
   st.session_state.conversation_rounds % SUMMARY_INTERVAL == 0 and \
   (not st.session_state.summaries or len(st.session_state.summaries) * SUMMARY_INTERVAL < st.session_state.conversation_rounds):
    with st.spinner("正在生成对话总结..."):
        start_index_for_summary = max(0, len(st.session_state.messages) - SUMMARY_INTERVAL)
        actual_messages_for_summary = st.session_state.messages[start_index_for_summary : len(st.session_state.messages)]
        summary_text = get_conversation_summary(actual_messages_for_summary)
        st.session_state.summaries.append(summary_text)
        st.toast(f"已为第 {st.session_state.conversation_rounds} 轮生成对话总结！")
        # 此处不需要重新运行

# --- 如果聊天为空，机器人初始问候 ---
if not st.session_state.messages and st.session_state.bots_in_chat:
    # 随机选择一个机器人，而不是总是第一个
    first_bot_name = random.choice(st.session_state.bots_in_chat)
    persona_details = st.session_state.bot_personas_data[first_bot_name]
    
    # 获取压缩记忆
    compressed_memory_text = cm.get_compressed_memory(first_bot_name)
    
    if compressed_memory_text:
        # 如果有压缩记忆，使用LLM基于压缩记忆生成随机问候
        prompt = (
            f"你是一个AI助手，为 {first_bot_name}（{persona_details['description']}）生成一个简短的聊天室开场问候。\n"
            f"基于以下长期记忆：\n{compressed_memory_text}\n\n"
            f"生成一个自然的、符合角色性格的问候语，长度不超过50字，反映角色的背景或记忆中的关键点。"
        )
        try:
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有创意的问候生成助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            initial_greeting = completion.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"生成 {first_bot_name} 的问候时出错：{e}")
            initial_greeting = persona_details.get("greeting", f"你好，我是 {first_bot_name}。")
    else:
        # 如果没有压缩记忆，使用persona.py中的预设问候
        initial_greeting = persona_details.get("greeting", f"你好，我是 {first_bot_name}。")
    
    timestamp = datetime.now()
    st.session_state.messages.append({"role": first_bot_name, "content": initial_greeting, "timestamp": timestamp})
    st.session_state.last_speaker = first_bot_name
    st.session_state.conversation_rounds += 1
    st.session_state.bot_memories[first_bot_name] = update_bot_memory(
        first_bot_name, persona_details, st.session_state.messages, st.session_state.bot_memories[first_bot_name]
    )
    st.rerun()


# --- 处理机器人自动对话逻辑 ---
if "auto_bot_turns" in st.session_state and st.session_state.auto_bot_turns > 0:
    if "current_auto_turn" not in st.session_state:
        st.session_state.current_auto_turn = 0
    
    if st.session_state.current_auto_turn < st.session_state.auto_bot_turns:
        with st.spinner(f"自动对话进行中... (第 {st.session_state.current_auto_turn + 1}/{st.session_state.auto_bot_turns} 轮)"):
            if bot_autonomous_turn():
                st.session_state.current_auto_turn += 1
                # 检查是否需要生成总结
                if st.session_state.conversation_rounds > 0 and \
                   st.session_state.conversation_rounds % SUMMARY_INTERVAL == 0 and \
                   (not st.session_state.summaries or len(st.session_state.summaries) * SUMMARY_INTERVAL < st.session_state.conversation_rounds):
                    with st.spinner("正在生成对话总结..."):
                        start_index_for_summary = max(0, len(st.session_state.messages) - SUMMARY_INTERVAL)
                        actual_messages_for_summary = st.session_state.messages[start_index_for_summary : len(st.session_state.messages)]
                        summary_text = get_conversation_summary(actual_messages_for_summary)
                        st.session_state.summaries.append(summary_text)
                        st.toast(f"已为第 {st.session_state.conversation_rounds} 轮生成对话总结！")
                st.rerun()
    else:
        # 自动对话完成，重置状态
        st.session_state.auto_bot_turns = 0
        st.session_state.current_auto_turn = 0
        st.toast("自动对话已完成！")
        st.rerun()
