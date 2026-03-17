"""
对话式 AutoML Web 应用 - 改进版

流程：
1. 思路生成 → 用户确认
2. 生成代码 → 自动执行
3. 语法检查 → 自动修复（最多3次）
4. 执行结果展示 → 数据下载
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.dialog_pipeline import DialogPipeline
from llm_client import get_llm_client, configure_llm

st.set_page_config(page_title="对话式 AutoML", page_icon="🤖", layout="wide")


class DialogAutoMLApp:
    def __init__(self):
        self._init_state()
    
    def _init_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'target_column' not in st.session_state:
            st.session_state.target_column = None
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None
        if 'llm' not in st.session_state:
            st.session_state.llm = None
        if 'current_thinking' not in st.session_state:
            st.session_state.current_thinking = None
        if 'pending_thinking_confirm' not in st.session_state:
            st.session_state.pending_thinking_confirm = None
    
    def render(self):
        st.title("🤖 对话式 AutoML")
        st.markdown("""
        **流程**: 思路生成 → 用户确认 → 代码自动执行 → 结果展示
        
        在每个阶段，我会先生成处理思路，您确认后再执行。
        """)
        
        self.render_sidebar()
        self.render_data_upload()
        
        if st.session_state.pipeline:
            self.render_chat()
        
        if st.session_state.pending_thinking_confirm:
            self.render_thinking_confirmation()
    
    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ 配置")
            
            api_key = st.text_input("API Key", type="password", value=os.getenv("LLM_API_KEY", ""))
            base_url = st.text_input("API Base URL", value="https://fast.poloai.top")
            model = st.text_input("模型", value="claude-sonnet-4-20250514-thinking")
            
            if st.button("保存配置"):
                if api_key:
                    configure_llm(base_url=base_url, api_key=api_key, model=model)
                    st.session_state.llm = get_llm_client()
                    st.success(f"✅ 已保存")
            
            st.divider()
            
            if st.session_state.pipeline:
                st.header("📊 状态")
                p = st.session_state.pipeline
                data_shape = f"{p.data.shape[0]}行 × {p.data.shape[1]}列" if p.data is not None else "无"
                st.metric("数据", data_shape)
                st.metric("目标", p.target_column or "无")
                st.metric("代码块", len(p.code_blocks))
                
                st.divider()
                
                if st.button("📥 导出代码"):
                    code = p.export_code()
                    st.session_state.generated_code = code
                
                if 'generated_code' in st.session_state:
                    st.text_area("代码", st.session_state.generated_code, height=150)
                
                if p.data is not None:
                    st.divider()
                    csv = p.get_data_download_link()
                    if csv:
                        st.download_button("📥 下载处理后数据", csv, "processed_data.csv", "text/csv")
            
            st.divider()
            if st.button("🔄 重新开始"):
                for k in list(st.session_state.keys()):
                    if k != 'llm':
                        del st.session_state[k]
                st.rerun()
    
    def render_data_upload(self):
        st.subheader("📁 数据上传")
        
        uploaded = st.file_uploader("上传数据", type=["csv", "xlsx"])
        
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            
            st.session_state.data = df
            
            st.session_state.target_column = st.selectbox(
                "目标列", list(df.columns), key="target_sel"
            )
            
            with st.expander("🎯 建模场景与需求（可选）"):
                st.session_state.modeling_scenario = st.text_area(
                    "描述您的建模场景和要求",
                    placeholder="例如：预测用户是否流失的二分类问题，需要重点关注召回率...",
                    key="scenario_input",
                    height=80
                )
            
            if st.button("✅ 加载数据"):
                self._init_pipeline()
    
    def _init_pipeline(self):
        if not st.session_state.llm:
            st.error("❌ 请先配置 LLM")
            return
        
        scenario = st.session_state.get("modeling_scenario", "")
        
        p = DialogPipeline(st.session_state.llm)
        result = p.load_data(st.session_state.data, st.session_state.target_column, scenario)
        
        if result["success"]:
            st.session_state.pipeline = p
            
            scenario_msg = ""
            if scenario:
                scenario_msg = f"""
**建模场景**: {scenario}
"""
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"""🎉 数据加载完成！{scenario_msg}
**数据信息**:
- 形状: {result['profile']['shape']}
- 数值特征: {len(result['profile']['numeric_columns'])}
- 类别特征: {len(result['profile']['categorical_columns'])}

您现在可以：
1. 说"清洗数据" - 进行数据清洗
2. 说"特征工程" - 创建新特征
3. 说"训练模型" - 训练模型

请选择下一步操作。"""
            })
    
    def render_chat(self):
        st.subheader("💬 对话")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("描述您的需求..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = self._handle_input(prompt)
                st.markdown(response["content"])
                
                if response.get("show_thinking"):
                    with st.expander("💡 处理思路"):
                        st.markdown(response["show_thinking"])
    
    def _handle_input(self, user_input: str) -> Dict:
        p = st.session_state.pipeline
        text = user_input.lower()
        
        # 数据清洗
        if "清洗" in text:
            if st.session_state.pending_thinking_confirm:
                return {"content": "请先确认当前的处理思路"}
            
            with st.spinner("🤔 生成处理思路..."):
                result = p.generate_cleaning_thinking(user_input)
            
            if result["success"]:
                st.session_state.pending_thinking_confirm = "cleaning"
                st.session_state.current_thinking = result["thinking"]
                
                return {
                    "content": "数据清洗思路已生成，请确认是否正确。",
                    "show_thinking": result["thinking"]
                }
        
        # 特征工程
        elif "特征" in text:
            if st.session_state.pending_thinking_confirm:
                return {"content": "请先确认当前的处理思路"}
            
            directions = self._extract_directions(user_input)
            
            with st.spinner("🤔 生成处理思路..."):
                result = p.generate_feature_thinking(directions)
            
            if result["success"]:
                st.session_state.pending_thinking_confirm = "feature"
                st.session_state.current_thinking = result["thinking"]
                
                return {
                    "content": "特征工程思路已生成，请确认是否正确。",
                    "show_thinking": result["thinking"]
                }
        
        # 模型训练
        elif "训练" in text or "模型" in text:
            if st.session_state.pending_thinking_confirm:
                return {"content": "请先确认当前的处理思路"}
            
            with st.spinner("🤔 生成处理思路..."):
                result = p.generate_model_thinking(user_input)
            
            if result["success"]:
                st.session_state.pending_thinking_confirm = "model"
                st.session_state.current_thinking = result["thinking"]
                
                return {
                    "content": "模型训练思路已生成，请确认是否正确。",
                    "show_thinking": result["thinking"]
                }
        
        else:
            return {
                "content": f"""📊 当前状态:
- 数据: {p.data.shape if p.data is not None else '无'}
- 目标: {p.target_column or '无'}
- 代码块: {len(p.code_blocks)}

可以说:
- "清洗数据" - 数据清洗
- "特征工程" - 创建特征
- "训练模型" - 训练模型"""
            }
    
    def render_thinking_confirmation(self):
        import logging
        logger = logging.getLogger('DialogApp')
        
        st.divider()
        st.subheader("⚠️ 确认处理思路")
        
        thinking = st.session_state.current_thinking
        st.markdown(f"**处理思路**: {thinking}")
        
        col1, col2 = st.columns(2)
        
        if col1.button("✅ 确认，执行代码"):
            logger.info("Confirm button clicked, calling _execute_current_stage")
            self._execute_current_stage()
        
        if col2.button("❌ 取消"):
            st.session_state.pending_thinking_confirm = None
            st.session_state.current_thinking = None
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "已取消"
            })
            st.rerun()
        
        st.divider()
    
    def _execute_current_stage(self):
        import logging
        logger = logging.getLogger('DialogApp')
        
        p = st.session_state.pipeline
        stage = st.session_state.pending_thinking_confirm
        
        logger.info(f"_execute_current_stage called, stage={stage}")
        
        with st.spinner("⚙️ 生成并执行代码..."):
            if stage == "cleaning":
                logger.info("Calling generate_cleaning_code...")
                result = p.generate_cleaning_code()
                logger.info(f"generate_cleaning_code returned: {result}")
            elif stage == "feature":
                result = p.generate_feature_code()
            elif stage == "model":
                result = p.generate_model_code()
            else:
                result = {"success": False, "message": "未知阶段"}
        
        st.session_state.pending_thinking_confirm = None
        st.session_state.current_thinking = None
        
        if result["success"]:
            # 更新数据
            if result.get("data_shape"):
                st.session_state.data = p.data
            
            msg = f"✅ {result['message']}"
            
            if result.get("data_shape"):
                msg += f"\n\n数据形状: {result['data_shape']}"
            
            # 添加消息到对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg
            })
            
            # 显示详细结果
            if result.get("result", {}).get("output_data") is not None:
                output_data = result["result"]["output_data"]
                if isinstance(output_data, pd.DataFrame):
                    with st.expander("📊 处理后数据预览"):
                        st.dataframe(output_data.head())
            
            # 显示详细输出
            exec_result = result.get("result", {})
            output_text = exec_result.get("output", "")
            if output_text and output_text.strip():
                if output_text.strip() != "执行完成":
                    with st.expander("📋 执行详情", expanded=True):
                        st.text(output_text)
            
            # 显示代码内容
            current_code = p.current_code
            if current_code:
                with st.expander("📝 执行的代码"):
                    st.code(current_code, language="python")
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ {result.get('message', '执行失败')}\n\n错误: {result.get('error', '')}"
            })
        
        st.rerun()
    
    def _extract_directions(self, text: str) -> list:
        dirs = []
        if "年龄" in text: dirs.append("创建年龄特征")
        if "面积" in text: dirs.append("创建面积特征")
        if "编码" in text: dirs.append("编码类别变量")
        if not dirs:
            dirs = ["创建基本特征", "编码类别变量"]
        return dirs


def main():
    app = DialogAutoMLApp()
    app.render()


if __name__ == "__main__":
    main()
