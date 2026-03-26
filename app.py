from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


# ----------------------------
# LLMから回答を取得する関数
# ----------------------------
def get_llm_response(input_text: str, expert_type: str) -> str:
    expert_prompts = {
        "A：マーケティング専門家": (
            "あなたはマーケティングの専門家です。"
            "市場分析、ターゲット設定、販促施策、SNS運用、ブランディングの観点から、"
            "わかりやすく実践的に回答してください。"
        ),
        "B：ITアーキテクト専門家": (
            "あなたはITアーキテクトの専門家です。"
            "システム設計、技術選定、保守性、拡張性、セキュリティの観点から、"
            "初心者にもわかるように具体的に回答してください。"
        ),
    }

    system_message = expert_prompts.get(
        expert_type,
        "あなたは親切で有能なアシスタントです。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{user_input}")
        ]
    )

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.predict(user_input=input_text)
    return response


# ----------------------------
# 画面設定
# ----------------------------
st.set_page_config(page_title="専門家AI相談アプリ", page_icon="🤖")
st.title("🤖 専門家AI相談アプリ")

# ----------------------------
# アプリ概要・操作方法
# ----------------------------
st.markdown("""
### アプリ概要
このアプリでは、入力したテキストをLLMに渡して回答を表示します。  
また、ラジオボタンで選んだ専門家の種類に応じて、AIの回答方針が変わります。

### 操作方法
1. 専門家の種類を選択してください  
2. 入力欄に質問内容を入力してください  
3. 「送信」ボタンを押してください  
4. 回答結果が画面に表示されます
""")

# ----------------------------
# APIキー確認
# ----------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY が設定されていません。環境変数を設定してください。")

# ----------------------------
# 入力フォーム
# ----------------------------
with st.form("input_form"):
    expert_type = st.radio(
        "専門家の種類を選んでください",
        ["A：マーケティング専門家", "B：ITアーキテクト専門家"]
    )

    input_text = st.text_area(
        "入力テキスト",
        placeholder="ここに質問を入力してください"
    )

    submitted = st.form_submit_button("送信")

# ----------------------------
# 回答表示
# ----------------------------
if submitted:
    if not input_text.strip():
        st.error("入力テキストを入力してください。")
    else:
        try:
            with st.spinner("回答を生成中です..."):
                answer = get_llm_response(input_text, expert_type)

            st.subheader("回答結果")
            st.write(answer)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")