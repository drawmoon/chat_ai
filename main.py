import streamlit as st

from bot import Bot

bot = Bot()


def new_chat():
    st.session_state.messages = []
    st.session_state.msg_id = 0

    global bot
    bot = Bot()


with st.sidebar:
    "[View the source code](https://github.com/drawmoon/chat_ai)"
    st.sidebar.button("新的聊天", on_click=new_chat)

if bot.prologue:
    st.chat_message("assistant").write(bot.prologue)

if user_input := st.chat_input("请输入你的问题..."):
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            placeholder = st.empty()
            full_message = ""

            def stream_write(c: str):
                global full_message
                full_message += c
                placeholder.write(full_message)

            bot.invoke_stream(user_input, lambda c: stream_write(c))
