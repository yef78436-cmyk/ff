import os
import time
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 加载环境变量
load_dotenv()

# ====================== 基础工具类 ======================
class ModelFactory:
    """模型工厂，统一管理所有大模型调用"""
    @staticmethod
    def get_llm(model_name: str, temperature: float = 0.7):
        if model_name.startswith("gpt"):
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
        elif model_name.startswith("deepseek"):
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL")
            )
        elif model_name.startswith("doubao"):
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("DOUBAO_API_KEY"),
                base_url=os.getenv("DOUBAO_BASE_URL")
            )
        else:
            raise ValueError(f"不支持的模型: {model_name}")

# ====================== 6个核心Agent ======================
class HotspotMonitorAgent:
    """Agent 1: 多平台热点监测Agent"""
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }

    def get_weibo_hotspots(self, limit: int = 10):
        """获取微博热搜"""
        try:
            response = requests.get("https://s.weibo.com/top/summary?cate=realtimehot", headers=self.headers)
            soup = BeautifulSoup(response.text, "html.parser")
            hotspots = []
            for item in soup.select(".td-02 a")[:limit]:
                hotspots.append(item.text.strip())
            return hotspots
        except Exception as e:
            print(f"获取微博热点失败: {e}")
            return []

    def get_douyin_hotspots(self, limit: int = 10):
        """获取抖音热点（简化版）"""
        try:
            response = requests.get("https://www.douyin.com/aweme/v1/web/hot/search/list/", headers=self.headers)
            data = response.json()
            hotspots = [item["word"] for item in data["data"]["word_list"][:limit]]
            return hotspots
        except Exception as e:
            print(f"获取抖音热点失败: {e}")
            return []

    def run(self, platforms: list = ["weibo", "douyin"], limit: int = 10):
        """运行热点监测"""
        all_hotspots = {}
        if "weibo" in platforms:
            all_hotspots["微博"] = self.get_weibo_hotspots(limit)
        if "douyin" in platforms:
            all_hotspots["抖音"] = self.get_douyin_hotspots(limit)
        return all_hotspots

class TopicPlanningAgent:
    """Agent 2: 选题策划Agent（长链推理）"""
    def __init__(self, llm):
        self.llm = llm

    def run(self, hotspots: dict, account_niche: str, history_hot_topics: list = None):
        """
        结合热点和账号定位生成选题
        :param hotspots: 热点字典
        :param account_niche: 账号定位（如"AI工具测评"、"职场干货"）
        :param history_hot_topics: 历史爆款选题
        """
        system_prompt = f"""
        你是专业的自媒体选题策划专家，擅长结合热点和账号定位生成高转化选题。
        账号定位：{account_niche}
        历史爆款选题：{history_hot_topics if history_hot_topics else "暂无"}
        
        请基于以下全网热点，生成本账号的5个差异化选题，要求：
        1. 选题必须结合账号定位，不能脱离领域
        2. 要有冲突感和好奇心，避免平淡
        3. 标注每个选题的预估热度和适合平台
        4. 每个选题附带100字左右的核心观点
        
        输出格式：
        1. 选题标题 | 预估热度：⭐⭐⭐⭐⭐ | 适合平台：小红书/抖音/公众号
           核心观点：xxx
        """
        
        human_prompt = f"全网热点：{hotspots}\n\n请生成5个高转化选题。"
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        return response.content

class MultiPlatformRewriteAgent:
    """Agent 3: 多平台内容改写Agent"""
    def __init__(self, llm):
        self.llm = llm
        self.platform_styles = {
            "xiaohongshu": """
            小红书风格要求：
            1. 标题用emoji开头，吸引眼球
            2. 多用短句，分段清晰，每段不超过3行
            3. 语气亲切，像和闺蜜聊天
            4. 结尾加相关话题标签（5-8个）
            5. 字数控制在500-800字
            """,
            "douyin": """
            抖音口播稿风格要求：
            1. 开头3秒必须有钩子，直击痛点
            2. 语速快，多用短句，避免长句
            3. 每15秒一个反转或干货点
            4. 结尾引导点赞关注
            5. 字数控制在300-500字（约1分钟）
            """,
            "gongzhonghao": """
            公众号文章风格要求：
            1. 标题要有深度，引发思考
            2. 结构清晰，有引言、正文、结论
            3. 内容详实，有案例和数据支撑
            4. 结尾有总结和互动
            5. 字数控制在1500-2000字
            """
        }

    def run(self, topic: str, core_view: str, platforms: list = ["xiaohongshu", "douyin", "gongzhonghao"]):
        """生成多平台内容"""
        results = {}
        for platform in platforms:
            system_prompt = f"""
            你是专业的{platform}内容创作者，请根据以下选题和核心观点，生成一篇符合{platform}风格的完整内容。
            {self.platform_styles[platform]}
            """
            
            human_prompt = f"选题：{topic}\n核心观点：{core_view}"
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            results[platform] = response.content
        return results

class FormattingAgent:
    """Agent 4: 排版适配Agent"""
    def run(self, content: str, platform: str):
        """根据平台自动排版"""
        if platform == "xiaohongshu":
            # 小红书排版：添加空行，优化emoji
            formatted = content.replace("\n", "\n\n")
            return formatted
        elif platform == "douyin":
            # 抖音排版：添加换行，方便口播
            lines = content.split("。")
            formatted = "\n".join([line.strip() + "。" for line in lines if line.strip()])
            return formatted
        elif platform == "gongzhonghao":
            # 公众号排版：添加标题层级
            formatted = content.replace("一、", "\n## 一、").replace("二、", "\n## 二、").replace("三、", "\n## 三、")
            return formatted
        return content

class PublishSchedulerAgent:
    """Agent 5: 发布调度Agent（预留接口）"""
    def __init__(self):
        self.best_times = {
            "xiaohongshu": ["07:30", "12:30", "18:30", "21:30"],
            "douyin": ["12:00", "18:00", "20:00", "22:00"],
            "gongzhonghao": ["08:00", "12:00", "18:00", "21:00"]
        }

    def get_best_publish_time(self, platform: str):
        """获取最佳发布时间"""
        now = time.strftime("%H:%M")
        for t in self.best_times[platform]:
            if t > now:
                return t
        return self.best_times[platform][0]

    def schedule_publish(self, content: str, platform: str, publish_time: str = None):
        """定时发布（需对接平台API）"""
        if not publish_time:
            publish_time = self.get_best_publish_time(platform)
        print(f"已调度 {platform} 内容于 {publish_time} 发布")
        # 这里对接对应平台的发布API
        return True

class DataAnalysisAgent:
    """Agent 6: 数据复盘Agent（预留接口）"""
    def run(self, platform: str, article_id: str):
        """获取文章数据并生成优化建议"""
        print(f"正在获取 {platform} 文章 {article_id} 的数据...")
        # 这里对接对应平台的数据API
        return "数据复盘建议：标题可以更有冲突感，增加案例数量"

# ====================== 主流程 ======================
class ContentMatrixAgent:
    """多平台内容矩阵生成主Agent"""
    def __init__(self, account_niche: str):
        self.account_niche = account_niche
        self.default_llm = ModelFactory.get_llm(os.getenv("DEFAULT_MODEL"))
        
        # 初始化所有子Agent
        self.hotspot_agent = HotspotMonitorAgent()
        self.topic_agent = TopicPlanningAgent(self.default_llm)
        self.rewrite_agent = MultiPlatformRewriteAgent(self.default_llm)
        self.formatting_agent = FormattingAgent()
        self.publish_agent = PublishSchedulerAgent()
        self.data_agent = DataAnalysisAgent()

    def run_full_workflow(self):
        """运行完整的内容生成工作流"""
        print("="*50)
        print("🚀 多平台内容矩阵生成Agent 启动")
        print(f"📌 账号定位：{self.account_niche}")
        print("="*50)

        # 1. 监测全网热点
        print("\n1. 正在监测全网热点...")
        hotspots = self.hotspot_agent.run(limit=15)
        print(f"✅ 获取到 {len(hotspots['微博'])} 条微博热点，{len(hotspots['抖音'])} 条抖音热点")

        # 2. 生成选题
        print("\n2. 正在生成高转化选题...")
        topics = self.topic_agent.run(hotspots, self.account_niche)
        print("✅ 选题生成完成：")
        print(topics)

        # 3. 选择一个选题生成多平台内容
        selected_topic = input("\n请输入你要生成内容的选题标题：")
        selected_core_view = input("请输入该选题的核心观点：")
        
        print("\n3. 正在生成多平台内容...")
        contents = self.rewrite_agent.run(selected_topic, selected_core_view)
        print("✅ 多平台内容生成完成")

        # 4. 自动排版
        print("\n4. 正在进行平台适配排版...")
        formatted_contents = {}
        for platform, content in contents.items():
            formatted_contents[platform] = self.formatting_agent.run(content, platform)
        print("✅ 排版完成")

        # 5. 输出结果
        print("\n" + "="*50)
        print("📄 生成结果预览：")
        print("="*50)
        
        for platform, content in formatted_contents.items():
            print(f"\n\n📱 {platform.upper()} 内容：")
            print("-"*30)
            print(content)

        # 6. 可选：定时发布
        publish = input("\n是否自动调度发布？(y/n)：")
        if publish.lower() == "y":
            for platform in formatted_contents.keys():
                self.publish_agent.schedule_publish(formatted_contents[platform], platform)

        print("\n🎉 全流程执行完成！")
        return formatted_contents

# ====================== 运行入口 ======================
if __name__ == "__main__":
    # 初始化Agent，修改为你的账号定位
    agent = ContentMatrixAgent(account_niche="AI工具测评与效率提升")
    
    # 运行完整工作流
    result = agent.run_full_workflow()
    
    # 保存结果到文件
    with open("content_result.txt", "w", encoding="utf-8") as f:
        for platform, content in result.items():
            f.write(f"{'='*50}\n")
            f.write(f"{platform.upper()} 内容\n")
            f.write(f"{'='*50}\n\n")
            f.write(content)
            f.write("\n\n\n")
    print("\n💾 结果已保存到 content_result.txt")
