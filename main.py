
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import streamlit as st
import plotly
import numpy as np
import glob
import string
from langdetect import detect, lang_detect_exception
import re


# Step-1: Read the files in the dataframe and apply the filter analysis window data
def filter_dataframe_by_date(data, start_date, end_date, column_name):
    data[column_name] = pd.to_datetime(data[column_name])
    filtered_data = data[data[column_name].between(start_date, end_date)]
    return filtered_data


# step-2: detect the smileys and emojis to convert to text
def convert_smileys_in_dataframe(df, column_name):
    smiley_dict = {
        ":)": "happy",          ":-)": "happy",
        ":D": "laughing",       ":-D": "laughing",
        ":(": "sad",            ":-(": "sad",
        ";)": "wink",           ";-)": "wink",
        ":P": "playful",        ":-P": "playful",
        ":O": "surprised",      ":-O": "surprised",
        "😍": "heart eyes",     "🔥": "fire",
        "👏": "clapping",       "😃": "happy",
        "😄": "happy",          "😁": "happy",
        "😆": "happy",          "😊": "happy",
        "😋": "happy",          "😎": "happy",
        "😜": "playful",        "😝": "playful",
        "😢": "sad",            "😭": "sad",
        "😉": "wink",           "😛": "wink",
        "😮": "surprised",      "😲": "surprised",
        "❤️": "heart",          "💔": "broken heart",
        "🙌": "celebration",    "🎉": "celebration",
        "🥳": "celebration",    "👍": "ok",
        "😂": "laugh out loud", "♥️": "love",
        "💪": "strong",         "💥": "fire",
        "🙏": "thanks",         "👐": "claps",
        "💞": "love"
    }

    pattern = r"(:-?\)|:-?D|:-?\(|;-?\)|:-?P|:-?O|😍|🔥|👏|😃|😄|😁|😆|😊|😋|😎|😜|😝|😢|😭|😉|😛|😮|😲|❤️|💔|🙌|🎉|🥳|👍|😂|♥️|💪|💥|🙏|👐|💞)"

    def replace_smiley(match):
        smiley = match.group()
        word = smiley_dict.get(smiley, smiley)

        return ' ' + word + ' '

    df[column_name] = df[column_name].apply(lambda x: re.sub(pattern, replace_smiley, x))
    return df

# step-3: remove unwanted emojis
def remove_emojis_from_text(text):
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F]|"  # emoticons
        r"[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
        r"[\U0001F680-\U0001F6FF]|"  # transport & map symbols
        r"[\U0001F1E0-\U0001F1FF]|"  # flags (iOS)
        r"[\U00002500-\U00002BEF]|"  # chinese char
        r"[\U00002702-\U000027B0]|"
        r"[\U000024C2-\U0001F251]|"
        r"[\U0001F926-\U0001F937]|"
        r"[\U00010000-\U0010FFFF]|"
        r"\u2640|\u2642|"
        r"\u2600-\u2B55|"
        r"\u200D|"
        r"\u23CF|"
        r"\u23E9|"
        r"\u231A|"
        r"\uFE0F|"
        r"\u3030",
        flags=re.UNICODE
    )

    return emoji_pattern.sub(r'', text)

#step-4: remove
def label_columns(df):
    # Initialize empty lists to store flag values for "bjp" and "ing"
    bjp_keywords = ['bjp', 'modi','Modi','MODI','பிஜேபி', 'மோடி']
    ing_keywords = ['congress', 'rahul', 'காங்கிரஸ்', 'ராகுல்']

    bjp_flags = []
    ing_flags = []

    # Convert keywords to lowercase for case-insensitive search
    bjp_keywords = [keyword for keyword in bjp_keywords]
    ing_keywords = [keyword for keyword in ing_keywords]

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        text = row['comment_textDisplay']

        # Check if any of the "bjp_keywords" are present in the text
        bjp_flag = 1 if any(keyword in text for keyword in bjp_keywords) else 0
        bjp_flags.append(bjp_flag)

        # Check if any of the "ing_keywords" are present in the text
        ing_flag = 1 if any(keyword in text for keyword in ing_keywords) else 0
        ing_flags.append(ing_flag)

    # Add the new columns "bjp" and "ing" to the DataFrame
    df['bjp'] = bjp_flags
    df['ing'] = ing_flags

    return df

#step-5: remove unlabeled records
def drop_rows_with_zeros(df):
    # Drop rows where both "bjp" and "ing" columns have 0 values
    df_filtered = df[(df['bjp'] != 0) | (df['ing'] != 0)]

    return df_filtered


#step-4: remove special characters in the text column
def remove_special_characters_and_return(df, column_name):
    # Define a regular expression pattern to match special characters
    special_char_pattern = r'[^a-zA-Z0-9\s]'

    # Remove special characters from the specified column in the DataFrame
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(special_char_pattern, '', str(x))
    )

    return df

#step-5: Drop NA's
def clean_and_trim_dataframe(df, column_name):
    # Trim whitespace from the values in the specified column
    df[column_name] = df[column_name].str.strip()

    # Drop rows with blank values in the specified column
    cleaned_df = df.dropna(subset=[column_name])

    return cleaned_df

#step-6: remove punctuations
def remove_punctuations_and_return(df, column_name):
    # Define a translation table to remove punctuations
    translation_table = str.maketrans('', '', string.punctuation)

    # Remove punctuations from the values in the specified column
    df[column_name] = df[column_name].apply(lambda x: x.translate(translation_table))

    return df

#step-7: remove duplicates
def remove_duplicates_and_return(df, column_name):
    # Remove duplicates from the specified column
    cleaned_df = df.drop_duplicates(subset=[column_name])

    return cleaned_df

# step-8: detect the language and add it as column
from langdetect import detect, DetectorFactory

# Set 'fallback' detection to prioritize more accurate results
DetectorFactory.seed = 0

def detect_language(df, text_column):
    # Mapping of ISO 639-1 codes to Indian language names
    indian_languages = {
        'hi': 'Hindi',
        'bn': 'Bengali',
        'te': 'Telugu',
        'ta': 'Tamil',
        'mr': 'Marathi',
        'ur': 'Urdu',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'or': 'Odia',
    }

    def detect_language_helper(text):
        try:
            detected_lang = detect(text)
            if detected_lang in indian_languages:
                return indian_languages[detected_lang], detected_lang
            return "English/Mixed", "en"
        except:
            return "unknown", "unknown"

    # Create new columns "language" and "language_code" using the detect_language_helper function
    df['language'], df['language_code'] = zip(*df[text_column].apply(detect_language_helper))
    return df

# step-2.1: detect the language unknown remove it
def remove_unknown_language(df):
    # Remove rows with 'unknown' language_code
    df_cleaned = df[df['language_code'] != 'unknown'].copy()

    return df_cleaned




# step-3: detect the language and transliterate
# from translate import Translator
#
# def translate_by_language(df):
#     def translate_text(text, lang_code):
#         try:
#             translator = Translator(from_lang=lang_code, to_lang='en')
#             translation = translator.translate(text)
#             return translation
#         except:
#             return "Translation Error"
#
#     df['translated_column'] = df.apply(lambda row: translate_text(row['comment_textDisplay'], row['language_code']), axis=1)
#     return df
#
# #step-4:
from googletrans import Translator

# def translate_comments_to_english(dataframe):
#     translator = Translator()
#
#     def translate_text(text, language_code):
#         try:
#             if pd.notnull(text):
#                 if len(text) > 500:  # Google Translate has a character limit, so we need to handle long texts
#                     chunks = [text[i:i+500] for i in range(0, len(text), 500)]
#                     translated_chunks = [translator.translate(chunk, dest='en', src=language_code).text for chunk in chunks]
#                     translated_text = " ".join(translated_chunks)
#                 else:
#                     translated_text = translator.translate(text, dest='en', src=language_code).text
#                 return translated_text
#             else:
#                 return text
#         except AttributeError:
#             return text
#
#     dataframe['Translated_Text'] = dataframe.apply(lambda row: translate_text(row['comment_textDisplay'], row['language_code']), axis=1)
#
#     return dataframe


from googletrans import Translator


def translate_comments_to_english(dataframe):
    translator = Translator()

    def translate_text(text, language_code):
        try:
            if pd.notnull(text):
                if len(text) > 500:
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    translated_chunks = [translator.translate(chunk, dest='en', src=language_code).text for chunk in chunks]
                    translated_text = " ".join(translated_chunks)
                else:
                    translated_text = translator.translate(text, dest='en', src=language_code).text
                return translated_text
            else:
                return text
        except (AttributeError, IndexError, Exception):
            return text

    dataframe['Translated_Text'] = dataframe.apply(lambda row: translate_text(row['comment_textDisplay'], row['language_code']), axis=1)

    return dataframe



# from deep_translator import (GoogleTranslator,
#                              ChatGptTranslator,
#                              MicrosoftTranslator,
#                              PonsTranslator,
#                              LingueeTranslator,
#                              MyMemoryTranslator,
#                              YandexTranslator,
#                              PapagoTranslator,
#                              DeeplTranslator,
#                              QcriTranslator,
#                              single_detection,
#                              batch_detection)
#
# def translate_and_add_column(df, source_column, language_code_column):
#     """
#     Translate the text in the source column based on the language code in the language code column
#     and add the translated text as a new column in the DataFrame.
#
#     Parameters:
#         df (pandas.DataFrame): The input DataFrame.
#         source_column (str): The name of the column containing the text to be translated.
#         language_code_column (str): The name of the column containing the language codes for each text.
#
#     Returns:
#         pandas.DataFrame: The input DataFrame with the translated text added as a new column.
#     """
#     # Function to translate text to English using GoogleTranslator
#     def translate_to_english(text, source_language_code):
#         return GoogleTranslator(source=source_language_code, target='en').translate(text)
#
#     # Create a new column 'Translated_Text' with the translated text
#     df['Translated_Text'] = df.apply(lambda row: translate_to_english(row[source_column], row[language_code_column]), axis=1)
#
#     return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_date = '2019-01-01'
    end_date = '2019-04-10'
    data = pd.read_csv("D:\\0_SHU_31018584\\Data\\Final_data\\youtube_apidata_34.csv", sep=',')
    count_row = data.shape[0]  # Gives number of rows
    count_col = data.shape[1]  # Gives number of columns
    print(count_row, count_col)
    data = filter_dataframe_by_date(data, start_date, end_date, "ytvideo_publishedAt")
    data = convert_smileys_in_dataframe(data, "comment_textDisplay")
    data['comment_textDisplay'] = data['comment_textDisplay'].apply(remove_emojis_from_text)
    data = label_columns(data)
    data = drop_rows_with_zeros(data)
    count_row = data.shape[0]  # Gives number of rows
    count_col = data.shape[1]  # Gives number of columns
    print(count_row, count_col)
    # data = remove_special_characters_and_return(data, "comment_textDisplay")
    data = clean_and_trim_dataframe(data, "comment_textDisplay")
    count_row = data.shape[0]  # Gives number of rows
    count_col = data.shape[1]  # Gives number of columns
    print(count_row, count_col)
    data = remove_punctuations_and_return(data, "comment_textDisplay")
    count_row = data.shape[0]  # Gives number of rows
    count_col = data.shape[1]  # Gives number of columns
    print(count_row, count_col)
    data = remove_duplicates_and_return(data, "comment_textDisplay")
    count_row = data.shape[0]  # Gives number of rows
    count_col = data.shape[1]  # Gives number of columns
    print(count_row, count_col)
    data = detect_language(data,text_column='comment_textDisplay')
    data = remove_unknown_language(data)
    count_row = data.shape[0]  # Gives number of rows
    count_col = data.shape[1]  # Gives number of columns
    print(count_row, count_col)
    # print(data[['comment_textDisplay']].tail(25))
    data1=data[['comment_textDisplay','bjp','ing', 'language', 'language_code']]
    data1.to_csv("D:\\0_SHU_31018584\\Data\\translated.csv", index=False)





    # data2 = translate_comments_to_english(data2)
    # # data2 = translate_and_add_column(data2, 'comment_textDisplay', 'language_code')
    # print(data2[['comment_textDisplay', 'Translated_Text']].head(10))
    # data2.to_csv("D:\\0_SHU_31018584\\Data\\translated.csv", index=False)












    # data2 = translate_comments_to_english(data2)
    # data2 = translate_by_language(data2)
    # data2['converted_column'] = data2.loc[data2['language_code'] == 'hi', 'comment_textDisplay'].apply(translate_to_english)
    # # Fill the 'converted_column' with original 'text' for non-Hindi rows
    # data2['converted_column'] = data2['converted_column'].fillna(data2['comment_textDisplay'])
    #
    # data3=data2[['comment_textDisplay', 'language_code', 'converted_column']]
    # print(data3[data3['language_code'] == 'hi'])

    # data3 = transliterate_to_english(data3, 'comment_textDisplay', 'language_code')
    # print(data2[['comment_textDisplay', 'language_code', 'converted_column']].tail(10))
    # print(data2[['comment_textDisplay','comment_textOriginal','language']].head(10))


