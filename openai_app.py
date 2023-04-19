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
  # https://docs.google.com/spreadsheets/d/1u_0afQiYZ-RN3-eaPC30EsALyY_gU7gNW62ZaGZu8go/edit
  workbook = gs.open_by_key(ss_key)
  
  return workbook



# ===========================================================================================================================================
#
# Home画面テキスト
#
# ===========================================================================================================================================
def home_text():
  
  
  st.write('', unsafe_allow_html=True)




# ===========================================================================================================================================
#
# ChatGPT
#
# ===========================================================================================================================================
def chatgpt():
  openai.api_key = chatgpt_api_Key
  
  question = st.text_area('▼ Question')
  st.write("")
  st.write("")
  
  if len(question) != 0:
    try:
      res = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "user", "content": question}
          ],
          temperature=1  # 温度（0-2, デフォルト1）
      )
      
      st.write('<h7>▼ Answer</h7>', unsafe_allow_html=True)
      st.write(res["choices"][0]["message"]["content"])
      
    except:
      st.error("回答を取得できませんでした。")
    
  elif len(question) == 0 or question == "":
      st.write('<h4><span style="color:#c0c0c0">質問を入力してください。</span></h4>', unsafe_allow_html=True)



# ===========================================================================================================================================
#
# ChatGPT（Memory）
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
                    inactive_color = '#c0c0c0', 
                    active_color="#1e90ff", 
                    track_color="#29B5E8"
                    )
  
  #st.write('<h4>Please enter text.</h4>', unsafe_allow_html=True)
  #st.write("")
  
  # **********
  # 通常モード
  # **********
  if toggle == False:    
    question = st.text_area('▼ Question')
    st.write("")
    st.write("")
    
    # 回答出力
    if len(question) != 0:
      # ChatGPT連携
      chatgpt_memory_class = ChatGPT_Memory()
      
      try:
        res = chatgpt_memory_class.chatgpt_chain.predict(human_input=question)
        
        st.write('<h7>▼ Answer</h7>', unsafe_allow_html=True)
        st.write(res)
        
      except:
        st.error("回答を取得できませんでした。")
      
    elif len(question) == 0 or question == "":
      st.write('<h4><span style="color:#c0c0c0">質問を入力してください。</span></h4>', unsafe_allow_html=True)
  
  # **********
  # 対話モード
  # **********
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
          
          if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
              message(st.session_state["generated"][i], key=str(i))
              message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
              
        except:
          st.error("メッセージを取得できませんでした。")
            
      elif len(question) == 0 or question == "":
        st.write('<h4><span style="color:#c0c0c0">質問を入力してください。</span></h4>', unsafe_allow_html=True)
        
    elif len(question) == 0 or question == "":
        st.write('<h4><span style="color:#c0c0c0">質問を入力してください。</span></h4>', unsafe_allow_html=True)



# ===========================================================================================================================================
#
# スプレッドシート連携
#
# ===========================================================================================================================================
def sheets_api():
  st.write('<h7><b><span style="color:#c0c0c0">スプレッドシートに「chatgpt@chatgpt-383502.iam.gserviceaccount.com」を共有してください。</span></b></h7>', unsafe_allow_html=True)
  st.write("")
  
  user_ss_key = st.text_input("▼ スプレッドシートキーの入力 （docs.google.com/spreadsheets/d/スプレッドシートキー/edit）")
  st.write("")
  
  if len(user_ss_key) != 0 or user_ss_key != "":
    
    try:
      workbook = spreadsheet_access(user_ss_key)
    
      worksheet_list = workbook.worksheets()
      
      worksheet_name_list = []
      for i in range(len(worksheet_list)):
        worksheet_name_list.append(worksheet_list[i].title)
      
      select_worksheet = st.selectbox("▼ シートを選択",worksheet_name_list)
      st.write("")
      
      worksheet = workbook.worksheet(select_worksheet)
      
      df = pd.DataFrame(worksheet.get_values()[1:], columns=worksheet.get_values()[0])
      
      st.write('<h7>▼ スプレッドシートデータ（読み込み）</h7>', unsafe_allow_html=True)
      st.dataframe(df)
      st.write("")
      
      button = st.button(label="Run")
      
      # ボタン押下
      if button:
        # データフレームをリスト化
        question_list = df.to_numpy().tolist()
        
        openai.api_key = chatgpt_api_Key
        
        i = 0
        for q in question_list:
          question = q[0]
          
          if len(question) != 0 and question != "":
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
      
    except:
      st.error("スプレッドシートキーが正しくありません。")



# ===========================================================================================================================================
#
# DeepL API
#
# ===========================================================================================================================================
def deepl_api():
  worksheet = workbook.worksheet("DeepL")
  deepl_auth_key = worksheet.acell('A2').value
  
  select_languach = st.selectbox("▼ Languach",["日本語 ⇒ 英語", "英語 ⇒ 日本語"])
  st.write("")
  st.write("")
  
  text = st.text_area("▼ Text")
  st.write("")
  st.write("")
  
  if len(text) != 0 or text != "":
    if select_languach == "日本語 ⇒ 英語":
      try:
        params = {
                    "auth_key": deepl_auth_key,
                    "text": text,
                    "source_lang": 'JA',
                    "target_lang": 'EN' 
                }
        # パラメータと一緒にPOSTする
        request = requests.post("https://api-free.deepl.com/v2/translate", data=params)
        result = request.json()
        
        st.write('<h7>▼ Result</h7>', unsafe_allow_html=True)
        st.write(result["translations"][0]["text"])
        
      except:
        st.error("翻訳できませんでした。")

    if select_languach == "英語 ⇒ 日本語":
      try:
        params = {
                    "auth_key": deepl_auth_key,
                    "text": text,
                    "source_lang": 'EN',
                    "target_lang": 'JA' 
                }
        # パラメータと一緒にPOSTする
        request = requests.post("https://api-free.deepl.com/v2/translate", data=params)
        result = request.json()
        
        st.write('<h7>▼ Result</h7>', unsafe_allow_html=True)
        st.write(result["translations"][0]["text"])
        
      except:
        st.error("翻訳に失敗しました。")
      
  elif len(text) == 0 or text == "":
        st.write('<h4><span style="color:#c0c0c0">テキストを入力してください。</span></h4>', unsafe_allow_html=True)



# ===========================================================================================================================================
#
# Whisper API
#
# ===========================================================================================================================================
def whisper_api():
  openai.api_key = chatgpt_api_Key
  
  audio_file = st.file_uploader("▼ Audio File", type=["mp3", "wav"])
  st.write("")
  st.write("")
  
  if audio_file != None: 
    audio_file_size = int(audio_file.size)
    if audio_file_size <= 25000000:
      submit_btn = st.button("Analyze")
      st.write("")
      st.write("")
      st.write("")
      
      if submit_btn:
        try:
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
# Image
#
# ===========================================================================================================================================
def image_create():
  openai.api_key = chatgpt_api_Key
  
  number_of_images = st.selectbox("▼ Number Of Images", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  st.write("")
  
  image_size = st.selectbox("▼ Create Image Size", ["1024x1024", "512x512", "256x256"])
  st.write("")
  
  order = st.text_area('▼ Order')
  st.write("")
  st.write("")
  st.write("")
  
  if len(order) != 0:
    try:
      res = openai.Image.create(
      
      prompt = order,
      n = number_of_images,
      size = image_size,
      response_format="b64_json",
      )
      
      st.write('▼ Create Image')
      
      #images_data = []
      for data, n in zip(res["data"], range(number_of_images)):
        img_data = base64.b64decode(data["b64_json"])
        # with open(f"image_{n}.png", "wb") as f:
        #     f.write(img_data)
            
        st.image(img_data)
        #images_data.append(img_data)
      
      # Download
      # if number_of_images == 1:
      #   st.download_button(label="Download", data=img_data, file_name="create_new_image.png")
      
      # else:
      #   st.download_button(label="Download", data=images_data, file_name="create_new_image.zip")
      #   pass
      
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
  ss_key = "13QH_0QLI57YSXp4_4O4CtaS5T7D2cOgNUnBmgaCuFf4"
  global workbook
  workbook = spreadsheet_access(ss_key)
  worksheet = workbook.worksheet("API Key")
  global chatgpt_api_Key
  chatgpt_api_Key = worksheet.acell('A2').value
  
  # Streamlit生成
  st.sidebar.write('<h1><span style="color:#f5deb3">OpenAI API</span> <span style="color:#c0c0c0">for</span></h1>', unsafe_allow_html=True)
  
  image = Image.open('images/old_kwm_logo.png')
  
  st.sidebar.image(image, use_column_width=True)
  st.sidebar.write("")
  st.sidebar.write("")
  
  # st.session_state.password = st.sidebar.text_input("▼ Password", type="password")
  
  # if st.session_state.password == "test01":
  
  # メニュー生成（bootstrap）
  # https://icons.getbootstrap.com/
  with st.sidebar:
    selected = option_menu("Menu", ["Home","ChatGPT", "ChatGPT（Memory）","SpreadSheet", "DeepL", "Whisper", "Image"], 
        icons=["house-door", "chat-dots", "chat-dots-fill", "file-earmark-spreadsheet", "translate", "volume-up", "image"], menu_icon="laptop", default_index=0)
    selected
  
  st.sidebar.write("")
  st.sidebar.write("")
  
  # メニュー選択
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
    st.write('<span style="color:#dcdcdc">ChatGPTの機能を利用できます。  \n こちらは単発のチャットになります。AIは前回の回答を記憶しません。  \n （消費トークン低）</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    chatgpt()
  
  # ChatGPT（Memory）
  elif selected == "ChatGPT（Memory）":
    st.write('<h1><span style="color:#f5deb3">ChatGPT（Memory）</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">ChatGPTの機能を利用できます。  \n ChatGPT（Memory）ではAIが前回の回答を記憶しており、会話を続けることが可能です。  \n （消費トークン高）</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    chat_mode()
    
  # SpreadSheet
  elif selected == "SpreadSheet":
    st.write('<h1><span style="color:#f5deb3">SpreadSheet</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">スプレッドシートとChatGpt APIを連携します。  \n スプレッドシートに入力した質問をChatGPTが回答します。（回答はスプレッドシート出力）</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    sheets_api()
    
  # DeepL
  elif selected == "DeepL":
    st.write('<h1><span style="color:#f5deb3">DeepL</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">DeepL APIを利用して、入力されたテキストを翻訳します。  \n ChatGPTは日本語よりも英語で質問した方が、回答精度が高いようです。</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    deepl_api()
    
  # Whisper
  elif selected == "Whisper":
    st.write('<h1><span style="color:#f5deb3">Whisper</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">Whisper APIを利用して、音声データを書き起こします。</span><b><span style="color:#e95464">（※変換可能サイズ：25MB）</span></b>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    whisper_api()
    
  # Image
  elif selected == "Image":
    st.write('<h1><span style="color:#f5deb3">Image</span></h1>', unsafe_allow_html=True)
    st.write('<span style="color:#dcdcdc">オーダーに合わせて、AIが画像生成・編集を行います。</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    
    image_create()