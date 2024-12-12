import streamlit as st
import json
import networkx as nx
from pyvis.network import Network
import tempfile
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, backref
from sqlalchemy.exc import NoResultFound
from json_repair import repair_json
from phi.tools.duckduckgo import DuckDuckGo
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from prompt import (
      PROMPT_EXPAND,
      PROMPT_DESCRIBE,
      PROMPT_DESCRIBE_SEARCH,
)
from jinja2 import Template
from dotenv import load_dotenv

load_dotenv()

# 初始化Assistant
assistant = Agent(
    model=OpenRouter(id="gpt-4o-mini"),
    response_format={"type": "json_object"},
    tools=[],
    markdown=False,
)

assistant_search = Agent(
    model=OpenRouter(id="gpt-4o-mini"),
    response_format={"type": "json_object"},
    tools=[DuckDuckGo()],
    markdown=False,
)

def call(prompt, search=False):
    tool = assistant_search if search else assistant
    while True:
        try:
            gen = tool.run(prompt, stream=False).content
            return json.loads(repair_json(gen))
        except Exception as e:
            print(f"生成时出错: {e}")
            # 可以根据需要添加重试逻辑或其他错误处理

# SQLAlchemy 设置
Base = declarative_base()

class Node(Base):
    __tablename__ = 'nodes'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, default="")
    parent_id = Column(Integer, ForeignKey('nodes.id'), nullable=True)
    # 正确设置自引用关系
    children = relationship(
        "Node",
        backref=backref('parent', remote_side=[id]),
        cascade="all, delete-orphan"
    )

# 创建 SQLite 引擎和会话
engine = create_engine('sqlite:///mind_map.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

class MindMap:
    def __init__(self):
        self.session = Session()
        try:
            # 尝试加载根节点
            self.root = self.session.query(Node).filter_by(parent_id=None).one()
        except NoResultFound:
            # 如果没有根节点，创建初始脑图
            self.root = Node(name="世界", description="一切开始的地方")
            self.session.add(self.root)
            self.session.commit()
            # 添加初始子节点
            initial_subnodes = {
            }
            for name, desc in initial_subnodes.items():
                child = Node(name=name, description=desc, parent=self.root)
                self.session.add(child)
            self.session.commit()
        self.current_node = self.root

    def delete_current_node(self):
        if self.current_node == self.root:
            st.warning("无法删除根节点。")
            return

        parent = self.current_node.parent
        self.session.delete(self.current_node)
        self.session.commit()
        self.current_node = parent
        st.success(f"节点已删除。当前导航到父节点：{self.current_node.name}")

    def get_children(self, node):
        return self.session.query(Node).filter_by(parent_id=node.id).all()

    def navigate_to(self, node_name):
        target = self.session.query(Node).filter_by(name=node_name, parent_id=self.current_node.id).first()
        if target:
            self.current_node = target
        else:
            st.warning("子节点不存在。")

    def navigate_to_node(self, node_name):
        """导航到指定名称的节点"""
        target = self.session.query(Node).filter_by(name=node_name).first()
        if target:
            self.current_node = target
        else:
            st.warning("节点不存在。")

    def expand_node(self, prompt, num_subjects=5):
        if not self.current_node:
            st.warning("此节点无法进一步扩展。")
            return

        parent_nodes = []
        node = self.current_node
        while node.parent:
            parent_nodes.append(node.parent.name)
            node = node.parent
        parent_node_names = " -> ".join(reversed(parent_nodes))

        prompt = Template(PROMPT_EXPAND).render(
            hint=prompt,
            current_node=self.current_node.name,
            parent_nodes=parent_node_names,
            num_subjects=num_subjects
        )
        suggestions = call(prompt, search=False)["subjects"]
        new_nodes = []
        for suggestion in suggestions:
            if suggestion and not self.session.query(Node).filter_by(name=suggestion, parent_id=self.current_node.id).first():
                new_node = Node(name=suggestion, description="", parent=self.current_node)
                self.session.add(new_node)
                new_nodes.append(suggestion)
        self.session.commit()
        if new_nodes:
            st.success("节点已扩展。新增子节点：" + ", ".join(new_nodes))
        else:
            st.info("没有新的子节点被添加。")

    def explain_node(self, search=False, prompt=""):
        if search:
            template = PROMPT_DESCRIBE_SEARCH
        else:
            template = PROMPT_DESCRIBE
        prompt = Template(template).render(
            subject=self.current_node.name,
            prompt=prompt,
        )
        try:
            explanation = call(prompt, search=search).get("explanation", "无法生成解释。")
            self.current_node.description = explanation
            self.session.commit()
            return explanation
        except Exception as e:
            st.error(f"生成解释时出错: {e}")
            return "无法生成解释。"

    def to_networkx(self):
        G = nx.DiGraph()
        def add_edges(node):
            children = self.get_children(node)
            for child in children:
                G.add_edge(node.name, child.name)
                add_edges(child)
        add_edges(self.root)
        return G

    def get_node_description(self, node):
        return node.description

    def get_full_map(self):
        nodes = self.session.query(Node).all()
        map_dict = {}
        for node in nodes:
            map_dict[node.name] = {
                "description": node.description,
                "children": [child.name for child in node.children]
            }
        return map_dict

def visualize_mindmap(mind_map):
    G = mind_map.to_networkx()
    net = Network(height='600px', width='100%', notebook=False, bgcolor='#1E1E1E', font_color='white')  # 深灰背景
    net.from_nx(G)
    net.toggle_physics(True)
    
    # 自定义节点和边的样式
    for node in net.nodes:
        node['color'] = {
            'background': '#FFFFFF',  # 节点背景为白色
            'border': '#4A4A4A',      # 节点边框为灰色
            'highlight': {'background': '#FFD700', 'border': '#FFD700'}  # 高亮时为金色
        }
    for edge in net.edges:
        edge['color'] = '#AAAAAA'  # 边颜色为浅灰色
    
    # 使用临时文件保存并加载
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    net.save_graph(tmp.name)
    with open(tmp.name, 'r', encoding='utf-8') as f:
        html = f.read()
    return html

def load_css():
    # 隐藏Streamlit默认菜单和页脚，并应用自定义样式
    hide_streamlit_style = """
            <style>
            /* 隐藏顶部菜单 */
            #MainMenu {visibility: hidden;}
            /* 隐藏分页器 */
            footer {visibility: hidden;}
            /* 隐藏加载动画 */
            .css-9s5bis {display: none;}
            /* 设置背景颜色 */
            body {
                background-color: #1E1E1E; /* 深灰背景 */
                color: #FFFFFF;
            }
            /* 自定义按钮颜色 */
            .stButton>button {
                background-color: #4A4A4A; /* 立体灰色按钮 */
                color: white;
            }
            .stTextArea textarea {
                background-color: #2B2B2B;
                color: white;
            }
            .stNumberInput input {
                background-color: #2B2B2B;
                color: white;
            }
            /* 自定义表单背景 */
            .streamlit-expanderHeader {
                color: #FFFFFF;
            }
            /* 卡片样式 */
            .card {
                background-color: #2A2A2A;
                padding: 20px;
                margin: 10px 0;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    # 设置页面配置，启用宽屏
    st.set_page_config(page_title="MindEden", layout="wide", page_icon="🧠")

    # 加载自定义CSS
    load_css()

    # 添加横幅
    banner_html = """
    <div class="card" style="background-color:#2A2A2A; padding:10px; border-radius:5px; text-align:center;">
        <h1 style="color:white;">MindEden 🧠</h1>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

    # 添加网站介绍
    introduction = """
    ## MindEden🧠
欢迎来到**MindEden**🧠。本平台旨在帮助您通过可视化的脑图结构来组织和扩展您的思维。您可以轻松地添加、浏览和管理节点，以便更好地规划您的项目、学习路径或任何其他需要结构化思维的任务。

**✨ 主要功能：**
- **📊 可视化脑图**：通过动态的脑图展示您的思维结构。
- **🧩 节点扩展**：使用智能提示自动生成并添加子节点。
- **📝 详细描述**：为每个节点添加详细的描述，帮助您更好地理解和规划。
- **🎨 专业设计**：简洁美观的界面设计，提升您的使用体验。
- **🚀 获取自由的钥匙**：在信息爆炸的时代，**获取自由的关键是打破信息封闭，打破信息隔阂**。
    """
    st.markdown(introduction, unsafe_allow_html=True)

    st.divider()

    st.title("MindEden")

    if 'mind_map' not in st.session_state:
        st.session_state.mind_map = MindMap()
    mind_map = st.session_state.mind_map

    # 使用列布局，左侧展示脑图，右侧展示控制面板
    hcol1, hcol2 = st.columns([4, 1])

    with hcol1:
        html = visualize_mindmap(mind_map)
        st.components.v1.html(html, height=600, scrolling=False)
 

    with hcol2:
        # 扩展节点卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("扩展当前节点")
        with st.form("expand_node_form"):
            prompt = st.text_area(
                "请输入用于扩展节点的提示（Prompt）",
                placeholder="例如：请列出获取网络资源的主要渠道和方法，并简要说明每种方法的优缺点。"
            )
            num_subjects = st.number_input(
                "生成子节点数量", 
                min_value=1, 
                max_value=20, 
                value=5
            )
            expand_submit = st.form_submit_button("扩展")
            if expand_submit:
                if prompt.strip() == "":
                    st.warning("提示（Prompt）不能为空。")
                else:
                    mind_map.expand_node(prompt.strip(), num_subjects=num_subjects)
                    st.rerun()

        st.subheader("删除当前节点")
        if st.button("🗑️ 删除"):
            confirm = st.warning("您确认要删除当前节点吗？此操作将删除所有子节点。", icon="⚠️")
            if st.button("确认删除"):
                mind_map.delete_current_node()
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # 显示当前路径
    paths = []
    node = mind_map.current_node
    while node:
        paths.insert(0, node.name)
        node = node.parent

    st.divider()
    
    # 当前路径卡片
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("当前路径: " + " -> ".join(paths))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

    if len(paths) > 1:
        if st.button("🔙 返回上一层"):
            mind_map.navigate_to_node(paths[-2])
            st.rerun()

    col1, col2 = st.columns([1, 3])

    with col1:
        # 子节点列表卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        children = mind_map.get_children(mind_map.current_node)
        if children:
            for child in children:
                if st.button(child.name):
                    mind_map.navigate_to(child.name)
                    if child.description == "":
                        mind_map.explain_node("请解释该节点")
                    st.rerun()
        else:
            st.write("此节点没有子节点。")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # 当前节点详细信息卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("当前节点详细信息:")
        st.markdown(f"### {mind_map.current_node.name}")
        st.markdown(mind_map.current_node.description)
        prompt = st.text_area(
            "请输入您对该节点的提问",
            placeholder="例如：请提供更多关于该节点的信息。"
        )
        if st.button("🔄 重新生成解释"):
            mind_map.explain_node(search=True, prompt=prompt)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
