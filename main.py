import streamlit as st
import streamlit_toggle as tog
from streamlit_option_menu import option_menu
from streamlit_chat import message

import openai

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials

from PIL import Image

import pandas as pd

import requests

import base64



# ===========================================================================================================================================
#
# スプレッドシートアクセス
#
# ===========================================================================================================================================
def spreadsheet_access(ss_key):
  # jsonファイルを使って認証情報を取得
  scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
  c = ServiceAccountCredentials.from_json_keyfile_name("chatgpt-383502-623f1357a143.json", scope)

  # 認証情報を使ってスプレッドシートの操作権を取得
  gs = gspread.authorize(c)

  # 共有したスプレッドシートのキーを使ってシートの情報を取得
  # https://docs.google.com/spreadsheets/d/13QH_0QLI57YSXp4_4O4CtaS5T7D2cOgNUnBmgaCuFf4/edit
  workbook = gs.open_by_key(ss_key)
  
  return workbook



# ===========================================================================================================================================
#
# Home画面テキスト
#
# ===========================================================================================================================================
def home_text():
  
  st.write('<h4><span style="color:#c0c0c0">このアプリケーションでは、OpenAIが提供している各種APIを連携しており、ChatGPTやWhisper等の機能を利用することが可能です。</span></h4>', unsafe_allow_html=True)
  st.write("")
  st.write("")
  st.write('<h4><span style="color:#c0c0c0">ご利用の際は、<span style="color:#e95464">社内ルールの遵守</span>をお願いします。</span></h4>', unsafe_allow_html=True)
  st.write('<h5><a href="https://kwm.kibe.la/notes/11961">【全社ルール・ガイドライン】生成AI（ChatGPT等）社内使用</a></h5>', unsafe_allow_html=True)
  st.write('<h1></h1>', unsafe_allow_html=True) # 調整用
  st.write('<h1></h1>', unsafe_allow_html=True) # 調整用
  st.write('<h1></h1>', unsafe_allow_html=True) # 調整用
  st.write('<h1></h1>', unsafe_allow_html=True) # 調整用
  
  # OpenAI ロゴ設定
  image = Image.open("images/openai_logo_toka.png")
  st.image(image, use_column_width=True)



# ===========================================================================================================================================
#
# ChatGPT
#
# ===========================================================================================================================================
@st.cache_resource
class ChatGPT_Memory:
  def __init__(self, ):

    self.template = """
     # Introduction
    - You are my exclusive professional female maid robot.
    - Please output the best result based on the following constraints

    # Constraints
    - Your answer must be in Japanese.
    - No important keywords are left out.
    - Keep the text concise.
    - If you cannot provide the best information, Let us know.

    {history}
    Human: {human_input}
    Assistant:
    """

    self.prompt = PromptTemplate(
      input_variables = ["history", "human_input"],
      template = self.template
    )

    self.chatgpt_chain = LLMChain(
      llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=chatgpt_api_Key),
      prompt = self.prompt,
      verbose = True,
      memory = ConversationBufferWindowMemory(k=2, memory_key="history")
    )



# ===========================================================================================================================================
#
# Chatモード
#
# ===========================================================================================================================================
def chat_mode():
  toggle = tog.st_toggle_switch(label="会話モード", 
                    key="Key1", 
                    default_value=False, 
                    label_after = False, 
                    inactive_color = "#c0c0c0", 
                    active_color="#1e90ff", 
                    track_color="#29B5E8"
                    )
  
  # ----------
  # 通常モード
  # ----------
  if toggle == False:    
    question = st.text_area("▼ Question")
    st.write("")
    st.write("")
    
    # 回答出力
    if len(question) != 0:
      # ChatGPT連携
      chatgpt_memory_class = ChatGPT_Memory()
      
      try:
        res = chatgpt_memory_class.chatgpt_chain.predict(human_input=question)
        
        # st.write('<h7><span style="color:#7fffd4">▼ Answer</span></h7>', unsafe_allow_html=True)
        # st.write(f'<span style="color:#7fffd4">{res}</span>', unsafe_allow_html=True)
        st.write('<h7>▼ Answer</h7>', unsafe_allow_html=True)
        st.write(res)
        
      except:
        st.error("回答を取得できませんでした。")
      
    elif len(question) == 0:
      st.write('<h4><span style="color:#c0c0c0">質問を入力してください。</span></h4>', unsafe_allow_html=True)
  
  # ----------
  # 対話モード
  # ----------
  else:
    if "generated" not in st.session_state:
      st.session_state.generated = []
    if "past" not in st.session_state:
      st.session_state.past = []
      
    with st.form("ChatGPT API"):
      question = st.text_area("▼ Question")
      submitted = st.form_submit_button("Send")
    
    # 「Send」ボタン押下
    if submitted:
      if len(question) != 0:
        chatgpt_memory_class = ChatGPT_Memory()
        
        try:
          res = chatgpt_memory_class.chatgpt_chain.predict(human_input=question)
          
          st.write("")
          st.write("")
          st.write('<h7>▼ Conversation</h7>', unsafe_allow_html=True)
          
          st.session_state.past.append(question)
          st.session_state.generated.append(res)
          
          if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])-1, -1, -1):
              message(st.session_state["generated"][i], key=str(i))
              message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
              
        except:
          st.error("メッセージを取得できませんでした。")
            
      elif len(question) == 0:
        st.write('<h4><span style="color:#c0c0c0">質問を入力してください。</span></h4>', unsafe_allow_html=True)
        
    elif len(question) == 0:
        st.write('<h4><span style="color:#c0c0c0">質問を入力してください。</span></h4>', unsafe_allow_html=True)



# ===========================================================================================================================================
#
# ChatGPT × スプレッドシート（会話記録／一括質問）
#
# ===========================================================================================================================================
def sheets_api():
  st.write("▼ Ready")
  st.write('<h7><b><span style="color:#c0c0c0">1. ご利用のスプレッドシートに、下記のアドレスを共有してください。</span></b>', unsafe_allow_html=True)
  
  st.markdown("""
              ```none
              chatgpt@chatgpt-383502.iam.gserviceaccount.com
              ```""")
  st.write('<h7><b><span style="color:#c0c0c0">2. 1行目にはヘッダーを設定してください。（例：A1「Human」 B1「AI」）</span></b></h7>', unsafe_allow_html=True)
  st.write("")
  
  # Mode
  select_mode = st.selectbox("▼ Mode", ["会話記録", "一括質問（単発）", "一括質問（連想）"])
  st.write("")
  
  user_ss_key = st.text_input("▼ SpreadSheet Key")
  gif = open("images/spreadsheet_key (cut).gif", "rb")
  contents = gif.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  gif.close()
  
  st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="spreadsheet key gif" width="700" height="65">',
    unsafe_allow_html=True,
    )

  st.write("")
  st.write("")
  
  if len(user_ss_key) != 0 :
    
    try:
      workbook = spreadsheet_access(user_ss_key)
    
      worksheet_list = workbook.worksheets()
      
      worksheet_name_list = []
      for i in range(len(worksheet_list)):
        worksheet_name_list.append(worksheet_list[i].title)
      
      select_worksheet = st.selectbox("▼ Sheet Name for Record",worksheet_name_list)
      st.write("")
      
      worksheet = workbook.worksheet(select_worksheet)
      
      # ---------
      # 会話記録
      # ---------
      if select_mode == "会話記録" :
        question = st.text_area("▼ Question")
        st.write("")
        st.write("")
        
        # 回答出力
        if len(question) != 0:
          # ChatGPT連携
          chatgpt_memory_class = ChatGPT_Memory()
          
          try:
            # 回答取得
            res = chatgpt_memory_class.chatgpt_chain.predict(human_input=question)
            
            # st.write('<h7><span style="color:#7fffd4">▼ Answer</span></h7>', unsafe_allow_html=True)
            # st.write(f'<span style="color:#7fffd4">{res}</span>', unsafe_allow_html=True)
            st.write('<h7>▼ Answer</h7>', unsafe_allow_html=True)
            st.write(res)
            
            str_list = list(filter(None, worksheet.col_values(1)))
            next_row = str(len(str_list) + 1)
            
            # スプレッドシート出力
            worksheet.update_cell(next_row, 1, question)  # 質問（Human）
            worksheet.update_cell(next_row, 2, res) # 回答（AI）
            
          except:
            st.error("回答の取得、または記録ができませんでした。")
            
        elif len(question) == 0:
          st.write('<h4><span style="color:#c0c0c0">質問を入力してください。</span></h4>', unsafe_allow_html=True)
        
      # ---------
      # 一括質問
      # ---------
      else:
        try:
          df = pd.DataFrame(worksheet.get_values()[1:], columns=worksheet.get_values()[0])
          
          st.write("<h7>▼ Read Data</h7>", unsafe_allow_html=True)
          st.dataframe(df)
          st.write("")
          
          button = st.button(label="Run")
          
          # -----
          # 単発
          # -----
          if select_mode == "一括質問（単発）":
            # ボタン押下
            if button:
              # データフレームをリスト化
              question_list = df.to_numpy().tolist()
              
              openai.api_key = chatgpt_api_Key
              
              i = 0
              for q in question_list:
                question = q[0]
                
                if len(question) != 0:
                  res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                      {"role": "user", "content": question}
                      ],
                    temperature=1  # 温度（0-2, デフォルト1）
                  )
                  
                  question_list[i][1] = (res["choices"][0]["message"]["content"])
                  
                else:
                  question_list[i][1] = "No Answer"
                  
                i = i + 1
              
              res_df = pd.DataFrame(question_list)
              
              set_with_dataframe(worksheet, res_df, row=2, include_column_header=False)

          # -----
          # 連想
          # -----
          elif select_mode == "一括質問（連想）":
            # ボタン押下
            if button:
              # データフレームをリスト化
              question_list = df.to_numpy().tolist()
              
              openai.api_key = chatgpt_api_Key
              
              i = 0
              first_question = ""
              next_question_str_1 = "私の最初の質問は「"
              next_question_str_2 = "」でしたが、"
              for q in question_list:
                question = q[0]
                
                if len(question) != 0:
                  # 初回限定
                  if i == 0:
                    first_question = question
                    res = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=[
                        {"role": "user", "content": question}
                        ],
                      temperature=1  # 温度（0-2, デフォルト1）
                    )
                  
                  # 2回目以降
                  else:
                    res = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=[
                        {"role": "user", "content": next_question_str_1 + first_question + next_question_str_2 +  question}
                        ],
                      temperature=1  # 温度（0-2, デフォルト1）
                    )
                    
                  question_list[i][1] = (res["choices"][0]["message"]["content"])
                  
                else:
                  question_list[i][1] = "No Answer"
                  
                i = i + 1
              
              res_df = pd.DataFrame(question_list)
              
              set_with_dataframe(worksheet, res_df, row=2, include_column_header=False)
               
        except:
          st.error("選択されたシートにはデータが存在しません。  \n 1行目はヘッダーを設定し、2行目以降に質問を入力してください。") 
        
    except:
      st.error("スプレッドシートキーが正しくありません。")



# ===========================================================================================================================================
#
# Whisper API
#
# ===========================================================================================================================================
def whisper_api():
  openai.api_key = chatgpt_api_Key

  # 音声ファイルアップロード
  audio_file = st.file_uploader("▼ Audio File", type=["mp3", "wav"])
  
  if audio_file != None: 
    audio_file_size = int(audio_file.size)
    convert_audio_file_size = round(audio_file_size / 1000000, 1)
    
    st.write('<h5><span style="color:#00bfff">取り込みファイルサイズ：' + str(convert_audio_file_size) + ' MB</span></h5>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    if audio_file_size <= 25000000:
      submit_btn = st.button("Analyze")
      st.write("")
      st.write("")
      st.write("")
      
      if submit_btn:
        try:
          # 書き起こし
          transcript = openai.Audio.transcribe("whisper-1", audio_file)
          st.write('<h7>▼ Text</h7>', unsafe_allow_html=True)
          st.write(transcript["text"])
          
        except:
          st.error("音声データの変換に失敗しました。")
          
    else:
      st.error("音声ファイルのサイズが制限の25MBを超えています。")
      
  else:
    st.write('<h4><span style="color:#c0c0c0">音声ファイルをアップロードしてください。</span></h4>', unsafe_allow_html=True)



# ===========================================================================================================================================
#
# DeepL API
#
# ===========================================================================================================================================
def translation(text, source_lang, target_lang):
  params = {
                    "auth_key": deepl_auth_key,
                    "text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang 
                }
  # パラメータと一緒にPOSTする
  request = requests.post("https://api-free.deepl.com/v2/translate", data=params)
  result = request.json()
  
  return result["translations"][0]["text"]


def deepl_api():
  worksheet = workbook.worksheet("DeepL")
  global deepl_auth_key
  deepl_auth_key = worksheet.acell("A2").value
  
  select_languach = st.selectbox("▼ Languach", ["日本語 ⇒ 英語", "英語 ⇒ 日本語"])
  st.write("")
  st.write("")
  
  text = st.text_area("▼ Text")
  st.write("")
  st.write("")
  
  if len(text) != 0:
    # 日本語 ⇒ 英語
    if select_languach == "日本語 ⇒ 英語":
      try:
        # 翻訳（日本語 ⇒ 英語）
        translation_text = translation(text, "JA", "EN")
        
        st.write('<h7>▼ English Text</h7>', unsafe_allow_html=True)
        st.write(translation_text)
        st.write("")
        st.write("")
        
        btn_chatgpt = st.button("ChatGPT")
        st.write("")
        st.write("")
        
        if btn_chatgpt:
          chatgpt_memory_class = ChatGPT_Memory()
      
          try:
            res = chatgpt_memory_class.chatgpt_chain.predict(human_input=translation_text)
            
            # 翻訳（英語 ⇒ 日本語）
            translation_res = translation(res, "EN", "JA")
            
            st.write('<h7>▼ ChatGPT Result</h7>', unsafe_allow_html=True)
            st.write(translation_res)
          
          except:
            st.error("メッセージを取得できませんでした。")
        
      except:
        st.error("翻訳に失敗しました。")

    # 英語 ⇒ 日本語
    if select_languach == "英語 ⇒ 日本語":
      try:
        # 翻訳（英語 ⇒ 日本語）
        translation_text = translation(text, "EN", "JA")
        
        st.write('<h7>▼ Japanese Text</h7>', unsafe_allow_html=True)
        st.write(translation_text)
        st.write("")
        st.write("")
        
        btn_chatgpt = st.button("ChatGPT")
        st.write("")
        st.write("")
        
        if btn_chatgpt:
          chatgpt_memory_class = ChatGPT_Memory()
      
          try:
            res = chatgpt_memory_class.chatgpt_chain.predict(human_input=translation_text)
            
            # 翻訳（日本語 ⇒ 英語）
            translation_res = translation(res, "JA", "EN")
            
            st.write('<h7>▼ ChatGPT Result</h7>', unsafe_allow_html=True)
            st.write(translation_res)
          
          except:
            st.error("メッセージを取得できませんでした。")
        
      except:
        st.error("翻訳に失敗しました。")
      
  elif len(text) == 0:
        st.write('<h4><span style="color:#c0c0c0">テキストを入力してください。</span></h4>', unsafe_allow_html=True)



# ===========================================================================================================================================
#
# Data Consideration
#
# ===========================================================================================================================================
def data_analysis():
  
  # ファイルアップロード
  file =st.file_uploader("▼ File Upload", type=["csv"])

  if file != None:
    try:
        df = pd.read_csv(file, encoding='cp932', engine="python")
        st.write("")
        st.write("")
        
        header_list = df.columns.values
        header_list_copy = header_list.copy()
        header_list_copy = header_list_copy.tolist()  # ndarray ⇒ list変換
        
        select_items = st.multiselect("▼ Select Item", header_list_copy)
        
        if len(select_items) >= 2: 
            df = df[select_items]
            st.write("▼ Create Data")
            st.write(df)
            
            pivot_indexes = st.multiselect("▼ Select Index", select_items)
            select_items_copy = select_items.copy()
            
            for del_index in pivot_indexes:
                select_items_copy.remove(del_index)
            
            pivot_values = st.multiselect("▼ Select Values", select_items_copy)
            
            aggfunc_items = ["カウント", "合計", "平均", "最大値", "最小値"]
            aggfunc_dic = {"カウント":"count", "合計":"sum", "平均":"mean", "最大値":"max", "最小値":"min"}
            povot_aggfuncs = st.multiselect("▼ Select Aggregation", aggfunc_items)
            st.write("")
            
            aggfunc_list = []
            for agg in povot_aggfuncs:
                aggfunc_list.append(aggfunc_dic[agg])
                
            if len(pivot_indexes) != 0 and len(pivot_values) != 0 and len(aggfunc_list) != 0:
              try:
                  df_pivot = pd.pivot_table(df, index=pivot_indexes, columns=None, values=pivot_values, aggfunc=aggfunc_list)
                  df_pivot.applymap("{:,.0f}".format)
                  
                  st.write("▼ Pivot Table")
                  st.dataframe(df_pivot)
                  st.write("")
                  st.write("")
                  
                  add_info = st.text_area("▼ Add Info")
                  st.write("")
                  
                  analysis_button = st.button("Analyse")
                  st.write("")
                  st.write("")
                  try:
                    if analysis_button:
                        openai.api_key = chatgpt_api_Key
                        res = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "Your answer must be in Japanese."},
                                {"role": "user", "content": df_pivot.to_json()},
                                {"role": "user", "content": "こちらのデータを分析してください。以下は追加情報になります。"},
                                {"role": "user", "content": add_info}
                            ],
                            temperature=1  # 温度（0-2, デフォルト1）
                        )
                        
                        st.write("▼ Result")
                        st.write(res["choices"][0]["message"]["content"])
                        st.write("")
                        st.write("（Total Token: " + str(res["usage"]["total_tokens"]) + "）")
                        
                  except:
                    st.error("分析に失敗しました。")
                      
              except:
                  st.error("ピボットテーブルの作成に失敗しました。")
                
    except:
        st.error("CSVファイルの読み込みに失敗しました。")



# ===========================================================================================================================================
#
# Image
#
# ===========================================================================================================================================
def image_create():
  openai.api_key = chatgpt_api_Key
  
  # 生成画像枚数
  number_of_images = st.selectbox("▼ Number Of Images", list(range(1, 6, 1)))
  st.write("")
  
  # 画像サイズ
  image_size = st.selectbox("▼ Image Size", ["1024x1024", "512x512", "256x256"])
  st.write("")
  
  # オーダー入力
  order = st.text_area('▼ Order')
  st.write("")
  st.write("")
  st.write("")
  
  if len(order) != 0:
    try:
      # 画像生成
      res = openai.Image.create(
      
      prompt = order,
      n = number_of_images,
      size = image_size,
      response_format="b64_json",
      )
      
      st.write("▼ Create Image")
      
      #images_data = []
      for data, n in zip(res["data"], range(number_of_images)):
        img_data = base64.b64decode(data["b64_json"])
            
        st.image(img_data)
      
    except:
      st.error("画像を生成できませんでした。")
    
  elif len(order) == 0 or order == "":
      st.write('<h4><span style="color:#c0c0c0">オーダーを入力してください。</span></h4>', unsafe_allow_html=True)



# ===========================================================================================================================================
#
# メイン処理
#
# ===========================================================================================================================================
#if __name__ == "__main__":
def openai_app_main():
  ss_key = "13QH_0QLI57YSXp4_4O4CtaS5T7D2cOgNUnBmgaCuFf4" # API Keyを管理しているシート
  global workbook, chatgpt_api_Key
  workbook = spreadsheet_access(ss_key)
  worksheet = workbook.worksheet("API Key")
  chatgpt_api_Key = worksheet.acell("A2").value
  
  # Streamlit生成
  st.sidebar.write('<h1><span style="color:#f5deb3">OpenAI API</span><span style="color:#c0c0c0"> for</span></h1>', unsafe_allow_html=True)
  
  # KWM ロゴ設定
  image = Image.open("images/kwm_logo.png")
  st.sidebar.image(image, use_column_width=True)
  st.sidebar.write("")
  st.sidebar.write("")
  
  # メニュー生成（bootstrap）
  # https://icons.getbootstrap.com/
  with st.sidebar:
    selected = option_menu("Menu", ["Home", "ChatGPT","SpreadSheet", "Whisper", "DeepL", "Data Analysis（CSV）", "Image"], 
        icons=["house-door", "chat-dots", "file-earmark-spreadsheet", "volume-up", "translate", "graph-up-arrow", "image"], menu_icon="laptop", default_index=0)
    selected
    
  st.sidebar.write("")
  
  # ------------
  # メニュー選択
  # ------------
  # Home
  if selected == "Home":
    st.write('<h1><span style="color:#f5deb3">Let’s Use OpenAI API!!</span></h1>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    home_text()
  
  # ChatGPT
  elif selected == "ChatGPT":
    st.write('<h1><span style="color:#f5deb3">ChatGPT</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">ChatGPTのベーシックな機能を利用することができます。  \n ※AIは1つ前までの回答を記憶しています。（消費トークン節約の為制限）</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    chat_mode()
    
  # SpreadSheet
  elif selected == "SpreadSheet":
    st.write('<h1><span style="color:#f5deb3">SpreadSheet</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">ChatGPT APIとスプレッドシートを連携します。会話記録、一括質問が可能です。  \n '
             'スプレッドシートの</span><b><span style="color:#00bfff"> A列を「質問用」</span></b>、<b><span style="color:#7fffd4">B列を「回答用」</span></b>とします。', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    sheets_api()
    
  # Whisper
  elif selected == "Whisper":
    st.write('<h1><span style="color:#f5deb3">Whisper</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">Whisper APIを利用して音声データを書き起こすことが可能です。  \n '
             '<span style="color:#e95464">（※書き起こし可能サイズ：25MB）</span></b>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    whisper_api()
    
  # DeepL
  elif selected == "DeepL":
    st.write('<h1><span style="color:#f5deb3">DeepL</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">DeepL APIを利用して、入力されたテキストを翻訳します。  \n '
             '翻訳したテキストをChatGPTに送ることも可能です。  \n '
             'ChatGPTは日本語よりも英語で質問した方が回答精度が高く、コストも抑えることができます。</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    deepl_api()
    
  # Data Analysis
  elif selected == "Data Analysis（CSV）":
    st.write('<h1><span style="color:#f5deb3">Data Analysis（CSV）</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">ChatGPTでCSVファイルのデータ分析を行います。  \n ' 
             'CSVファイルのデータ数が多い場合は、トークンの上限に達する可能性が高い為、  \n <b><span style="color:#7fffd4">画面上でピボットテーブルを作成</span></b>していきます。</span>  \n '
             '<b><span style="color:#e95464">※トークンが上限に達した場合は分析することができません。</span></b>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    data_analysis()
    
  # Image
  elif selected == "Image":
    st.write('<h1><span style="color:#f5deb3">Image</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">オーダーに合わせて、AIが画像の生成を行います。  \n 生成した画像は、右クリック ⇒「名前を付けて画像を保存」でダウンロードできます。</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    image_create()
