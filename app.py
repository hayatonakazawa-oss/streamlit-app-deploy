from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# 画面設定
# ----------------------------
st.set_page_config(page_title="専門家AI相談アプリ", page_icon="🤖")
st.title("🤖 専門家AI相談アプリ")

# ----------------------------
# アプリ概要・操作方法の表示
# ----------------------------
st.markdown("""
### アプリ概要
このWebアプリでは、入力したテキストをLLMに渡して回答を取得できます。  
さらに、ラジオボタンで選んだ専門家タイプに応じて、AIの振る舞いを切り替えます。

### 操作方法
1. 専門家の種類を選択してください  
2. 入力欄に質問や相談内容を入力してください  
3. 「送信」ボタンを押してください  
4. 回答結果が画面下に表示されます
""")

# ----------------------------
# LLMから回答を取得する関数
# 引数:
#   input_text: ユーザー入力
#   expert_type: ラジオボタンの選択値
# 戻り値:
#   LLMの回答文字列
# ----------------------------
def get_llm_response(input_text: str, expert_type: str) -> str:
    expert_prompts = {
        "マーケティング専門家": (
            "あなたはマーケティングの専門家です。"
            "市場分析、ターゲット設定、販促施策、SNS運用、ブランディングの観点から、"
            "わかりやすく実践的にアドバイスしてください。"
        ),
        "ITアーキテクト専門家": (
            "あなたはITアーキテクトの専門家です。"
            "システム設計、技術選定、保守性、拡張性、セキュリティの観点から、"
            "わかりやすく具体的にアドバイスしてください。"
        ),
        "キャリア相談専門家": (
            "あなたはキャリア相談の専門家です。"
            "相談者の状況を整理し、選択肢・強み・次の一歩が明確になるように、"
            "丁寧で前向きにアドバイスしてください。"
        ),
    }

    system_message = expert_prompts.get(
        expert_type,
        "あなたは親切で有能なアシスタントです。"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{user_input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    chain = prompt | llm
    response = chain.invoke({"user_input": input_text})

    return response.content


# ----------------------------
# APIキー確認
# ----------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY が設定されていません。環境変数を設定してください。")

# ----------------------------
# 入力フォーム
# ----------------------------
with st.form("question_form"):
    expert_type = st.radio(
        "専門家の種類を選んでください",
        ["マーケティング専門家", "ITアーキテクト専門家", "キャリア相談専門家"],
        horizontal=True
    )

    input_text = st.text_input("入力テキスト", placeholder="例：新サービスの集客方法を考えてください")

    submitted = st.form_submit_button("送信")

# ----------------------------
# 回答表示
# ----------------------------
if submitted:
    if not input_text.strip():
        st.error("入力テキストを入力してください。")
    else:
        with st.spinner("LLMが回答を生成中です..."):
            try:
                answer = get_llm_response(input_text, expert_type)
                st.subheader("回答結果")
                st.write(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")