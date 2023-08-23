"""
Sentiment Analysis using YouTube Comments - Prime Minister Election 2019
Author: Jagadeesan Rajalakshmi Vellaichamy
Reviewer: Dani Papamaximou
Created At: 20/08/2023
"""

#Import the necessary python libraries
# pip install pandas
# pip install glob
# pip install nltk
# pip install textblob
# pip install scikit-learn
# pip install transformers
# pip install torch
# pip install tqdm
# pip install numpy
# pip install langdetect
import pandas as pd
import glob
import string
import re
import numpy as np
import nltk
import torch
from nltk.corpus import stopwords
from textblob import Word
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from langdetect import detect, DetectorFactory

import warnings
warnings.filterwarnings("ignore")
##########################################################################################
#Step1: Read files from source directory
def FileReadFromDirectory(FileDirectory, FilePattern):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function reads the files present in a directory based on search pattern in files
    :param FileDirectory: The location of the directory
    :type FileDirectory: str
    :param FilePattern: The file pattern required
    :type FilePattern: str
    :return: dataframe
    """
    FilesList = glob.glob(FileDirectory + FilePattern)
    print(FilesList)
    dataframes = []
    for filename in FilesList:
        Youtube_Comments = pd.read_csv(filename, sep=',')
        dataframes.append(Youtube_Comments)
        Youtube_Comments = pd.concat(dataframes, ignore_index=True)
    return Youtube_Comments

#Step2: Filter the dataframe based on date filters
def AnalysisWindowTimePeriodFilter(raw_date, start_date, end_date, column_name):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function applies the date filter based on the analysis window required
    :param data: The data frame
    :type data: data frame
    :param start_date: The start date of the analysis window
    :type start_date: str
    :param end_date: The end date of the analysis window
    :type end_date: str
    :param column_name: The name of the column where the filter should be applied
    :type column_name: str
    :return: data frame
    """
    raw_date[column_name] = pd.to_datetime(raw_date[column_name])
    raw_date['PublishDate'] = raw_date[column_name].dt.strftime('%d-%m-%Y')
    raw_date['PublishWeek'] = raw_date[column_name].dt.strftime('%U')
    raw_date['PublishMonth'] = raw_date[column_name].dt.strftime('%m')
    raw_date['PublishYear'] = raw_date[column_name].dt.strftime('%Y')
    raw_date['PublishMonthYear'] = raw_date[column_name].dt.strftime('%b%Y')
    raw_date['PublishHour'] = raw_date[column_name].dt.strftime('%H')
    datefilter = raw_date[raw_date[column_name].between(start_date, end_date)]
    return datefilter
###############################################################################################
#Step3: Convert needed Emoji and Smiley to text
def SmileyConversiontoTexts(SmileytoTextdf, column_name):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function converts useful smileys to text
    :param SmileytoTextdf: The data frame
    :type SmileytoTextdf: data frame
    :param column_name: The text column which has YouTube comments
    :type column_name: str
    :return: data frame
    """
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

    def smileytotext(match):
        smiley = match.group()
        word = smiley_dict.get(smiley, smiley)

        return ' ' + word + ' '

    SmileytoTextdf[column_name] = SmileytoTextdf[column_name].apply(lambda x: re.sub(pattern, smileytotext, x) if isinstance(x, str) else x)
    return SmileytoTextdf
###############################################################################################
#Step4: Remove irrelevant smileys from text column
def EmojiRemovalfromComments(comments):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes unwanted smileys and emojis
    :param comments: The comments text
    :type comments: str
    :return: str
    """
    if isinstance(comments, str):
        smileyemoji_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  
                                  u"\U0001F300-\U0001F5FF"  
                                  u"\U0001F680-\U0001F6FF"  
                                  u"\U0001F700-\U0001F77F"  
                                  u"\U0001F780-\U0001F7FF"  
                                  u"\U0001F800-\U0001F8FF"  
                                  u"\U0001F900-\U0001F9FF"  
                                  u"\U0001FA00-\U0001FA6F"  
                                  u"\U0001FA70-\U0001FAFF"  
                                  u"\U00002702-\U000027B0"  
                                  u"\U000024C2-\U0001F251"
                                  "]+", flags=re.UNICODE)
        return smileyemoji_pattern.sub(r'', comments)
    else:
        return comments

###############################################################################################
#Step5: Remove the text with NAs
def Remove_NAs_Blanks(sourcedata, columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes comments which are having NA, blanks
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :return: data frame
    """
    sourcedata[columnname] = sourcedata[columnname].str.strip()
    trimmed_df = sourcedata.dropna(subset=[columnname])
    return trimmed_df
###############################################################################################
#Step6: Punctuations removal in comments
def Punctuations_Removal(sourcedata, comments_column):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes punctuations in comments
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :param comments_column: The comments text column
    :type comments_column: str
    :return: data frame
    """
    translation_table = str.maketrans('', '', string.punctuation)
    sourcedata[comments_column] = sourcedata[comments_column].apply(lambda x: x.translate(translation_table))
    return sourcedata
###############################################################################################
#Step7: Duplicates removal in comments
def DuplicateCommentsRemoval(sourcedata, columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes duplicate comments
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :return: data frame
    """
    nodupdf = sourcedata.drop_duplicates(subset=[columnname])
    return nodupdf
###############################################################################################
#Step8: Identify the regional language based on text comments
DetectorFactory.seed = 0
def Language_Identification(sourcedata, columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function identifies the regional languages in the comments column
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :return: data frame
    """
    indi_lang = {
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
        'or': 'Odia'
    }

    def Language_Identification_helper(comments):
        try:
            detected_lang = detect(comments)
            if detected_lang in indi_lang:
                return indi_lang[detected_lang], detected_lang
            return "English", "en"
        except:
            return "unknown", "unknown"

    sourcedata['language'], sourcedata['language_code'] = zip(*sourcedata[columnname].apply(Language_Identification_helper))
    return sourcedata
###############################################################################################
#Step9: Remove the comments with language unknown or not identified
def Unidentified_language_removal(sourcedata):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes unknown languages
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :return: data frame
    """
    # Remove rows with 'unknown' language_code
    validlangdf = sourcedata[sourcedata['language_code'] != 'unknown'].copy()
    return validlangdf
###############################################################################################
#Step10: Remove the comments with single grams
def SinglegramComments_Removal(sourcedata, columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes the comments with only 1 word in it
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :return: data frame
    """
    sourcedata = sourcedata[sourcedata[columnname].str.split().str.len() > 1]
    return sourcedata
###############################################################################################
#Step11: Remove the numbers in the comments
def NumbersinComments_Removal(sourcedata, columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes the numbers in the comments
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :return: data frame
    """
    sourcedata[columnname] = sourcedata[columnname].apply(lambda x: re.sub(r'\d+', '', x))
    return sourcedata
###############################################################################################
#Step12: Remove the repeat words in the comments
def RepeatwordsInCommentsRemoval(sourcedata, columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes the repeated words in the comments
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :return: data frame
    """
    sourcedata[columnname] = sourcedata[columnname].apply(lambda x: ' '.join(dict.fromkeys(x.split())))
    return sourcedata
###############################################################################################
#Step13: Additional iteration in updating the Roman Script of Indian regional languages
#List of custom words (bag of words) used in identifying the Roman script of indian languages
words_to_check = {
    'Hindi': ['aap','hum','yeh','sur','nat','bhi','jee','koi','aao','kya','maa','har','nit','bal','hai','din','kal','man','mai','tum','dil','mel','bol','hal','aur','kab','ban','hun','lev','hua','dom','bas','lou','kar','mat','dam','nas','nav','dut','gam','dev','rah','git','ram','ras','roz','laal','maaf','naam','raat','jald','maan','paas','rahi','saaf','aage','nach','rais','taap','gyan','gair','maya','dard','hona','jana','upar','liye','mana','chod','prem','band','chal','nayi','bhag','tark','chah','jiye','kuch','patr','tele','kadi','tika','atma','hand','hara','naav','pata','bojh','daak','sang','suru','daal','kaam','bhav','mukh','baat','jaag','urja','baja','dand','hans','nahi','path','dhua','nari','bali','lohe','loka','loni','vrat','jyon','mani','naak','sham','noor','mouj','waqt','zila','chor','kavi','khel','sima','deta','khub','soch','dhan','naya','dukh','lagi','nira','doga','lahu','pani','ekta','data','pita','garv','ghar','mera','desh','teji','raja','roop','rang','haar','kone','gaadi','jaisa','karna','paani','saath','uchit','bheed','garmi','karne','naari','taana','vastu','yatra','dhyan','garam','jaldi','karta','laakh','maang','udyog','khush','chaya','kadam','kuchh','niyam','pyaar','sagar','aankh','aaram','gayak','nayak','parya','yuddh','gyaan','mitti','samay','tarah','cinta','tatha','andar','divas','akeli','chota','bhakt','pauna','satya','jivan','kursi','saneh','avsar','mooch','paida','dalne','janam','kshan','odhni','pankh','upyog','daman','keeda','palna','badan','dhire','lakar','lagta','bagal','hathi','manch','poora','bahut','lagna','namak','varan','jevan','naada','vastr','badal','dhuaa','vidhi','humre','baste','jiwan','jadoo','basti','baten','navin','kabhi','beech','chand','kanha','nipat','bhaav','kajal','bhara','karya','katha','munde','bhool','murti','zarur','mudit','sidhi','daana','khaas','kisan','naadi','khoob','konsa','kiran','nidhi','nanha','sthan','cheta','lajja','paksh','kadar','lamba','patra','dagar','farak','patth','maarg','karan','mahal','khata','takat','kheli','dhaar','khana','tirth','ghoos','khyal','dhatu','goonj','treta','dhood','ruchi','dhool','tukda','haath','sadaa','tyaag','antib','bilkul','dheere','taakat','yahaan','zaroor','chehra','humein','laayak','chetan','saahas','vichar','zubaan','bhasha','takkal','vahaan','chinta','dekhna','sanket','vigyan','dimaag','hansna','sanyog','virodh','makaan','sansay','ashaat','mausam','chupke','vritti','nagari','pallav','unchai','atithi','jalana','nikhar','dharna','haraan','sangam','baccha','hamare','khayal','sanyam','janlev','samaaj','vastav','prabha','baatna','jhapat','lashan','prerna','dhvani','sankat','bahana','dhyaan','vishay','choona','nashta','preeti','sapnaa','vyakti','dhakka','purush','shakti','kahani','shanti','bhajan','kaamna','shreya','yantra','katori','sharir','kavita','keemat','bhojan','khelna','zoorna','kudrat','sparsh','dhoodh','doosra','nirnay','spasht','sundar','daaman','kamaal','nirmal','swapna','kamzor','swasth','dastak','paudha','gathri','peedit','mahila','prayas','swayam','gaurav','prakop','khidki','dharam','raksha','toofan','kirana','rupaya','sachet','rupiya','chahiye','vaastav','achchha','zindagi','hungama','chalana','sandesh','vinamra','koshish','macchar','nivedan','vishram','vishesh','bhashan','duskami','drishya','sacchai','uplabdh','dheeraj','patthar','pragati','sanyasi','vasudha','bandish','barasna','sankhya','bandhna','pradaan','vimarsh','pradesh','santaan','dilwala','vishwas','bhagwan','chetana','vyanjan','chintan','mulayam','bhushan','bhraman','sindoor','chakkar','nischay','nirdesh','pakshap','swabhav','pichhda','prakash','prerana','prishth','dhaaran','dharati','trishna','triveni','uddeshya','parchhai','chutkara','santulan','kvyapaar','samjhana','jhanjhna','dikhlana','prayatna','shikayat','vyavahar','shradhha','kartavya','siddhant','dakshina','bikharna','charitra','pahunche','suraksha','paryatan','taiyaari','tatkalin','ghinouna','parvachan','vichchhed','chopadiya','dhaaranaa','baksheesh','sangharsh','sanrachna','vyavastha','nishpatti','chikitsak','sindhutva','dhakelana','giraftaar','dhanyavaad','niyantraan','pratishodh','swatantrata','pratiyogita','pratispardha'],
    'Bengali': ['alo','ase','din','maa','nai','nei','noy','paa','ami','eto','kya','koi','ato','eta','jao','mar','rup','sei','tui','abar','aati','ache','anek','baal','boli','bose','chai','didi','dike','emon','haan','haat','habe','hobe','jabo','jana','kaaj','keno','kore','kuno','lage','lali','mama','mane','mone','naam','naki','nijo','onno','pujo','saja','suru','vaat','asbe','boro','haoa','pora','saho','thik','amar','tumi','paro','shob','taai','koto','balo','kaal','toke','baba','chul','ghar','hare','jabe','kono','koro','mata','mere','mile','more','moto','name','onek','opor','pare','pele','rate','rong','acche','aasha','achhe','adhik','baaje','bhalo','bhora','chaai','dekhe','dhoro','email','holam','karon','khela','kichu','kotha','lomba','matha','porbe','raate','roilo','snaan','tomay','varsa','ashon','ashte','ashun','bhebe','bhule','chaay','gache','korbe','lagbe','rakho','ekbar','korte','kemon','aache','bolte','tomar','jemon','kemne','kamon','parbe','amake','chele','choto','hashe','kheye','khete','khusi','lojja','mayer','natok','pashe','patha','phire','shuru','thake','tomra','aadhar','aamaar','ananda','ashaay','bhasha','britha','chaalo','chhoto','chokhe','deoyal','gobhir','saathe','avabar','bondhu','hochhe','shomoy','korcho','shathe','bujhte','lagche','kobita','bilkul','dheere','taakat','yahaan','zaroor','chehra','humein','laayak','chetan','saahas','vichar','zubaan','takkal','vahaan','chinta','dekhna','sanket','vigyan','dimaag','hansna','sanyog','virodh','makaan','sansay','ashaat','mausam','chupke','vritti','nagari','pallav','unchai','atithi','jalana','nikhar','dharna','haraan','sangam','baccha','hamare','khayal','sanyam','janlev','samaaj','vastav','prabha','baatna','jhapat','lashan','prerna','dhvani','sankat','bahana','dhyaan','vishay','choona','nashta','preeti','sapnaa','vyakti','dhakka','purush','shakti','kahani','shanti','bhajan','kaamna','shreya','yantra','katori','sharir','kavita','keemat','bhojan','khelna','zoorna','kudrat','sparsh','dhoodh','doosra','nirnay','spasht','sundar','daaman','kamaal','nirmal','swapna','kamzor','swasth','dastak','paudha','gathri','peedit','mahila','prayas','swayam','gaurav','prakop','khidki','dharam','raksha','toofan','kirana','rupaya','sachet','rupiya','apnake','ashena','bangla','dekhte','jibone','school','shudhu','tahole','thakbe','tokhon','tomake','aananda','krishno','opekkha','thaakbe','bhushon','korecho','bujhchi','chahiye','vaastav','achchha','zindagi','hungama','chalana','sandesh','vinamra','koshish','macchar','nivedan','vishram','vishesh','bhashan','duskami','drishya','sacchai','uplabdh','dheeraj','patthar','pragati','sanyasi','vasudha','bandish','barasna','sankhya','bandhna','pradaan','vimarsh','pradesh','santaan','dilwala','vishwas','bhagwan','chetana','vyanjan','chintan','mulayam','bhushan','bhraman','sindoor','chakkar','nischay','nirdesh','pakshap','swabhav','pichhda','prakash','prerana','prishth','dhaaran','dharati','trishna','triveni','lallike','nainaki','urevalo','uddeshya','parchhai','chutkara','santulan','kvyapaar','samjhana','jhanjhna','dikhlana','prayatna','shikayat','vyavahar','shradhha','kartavya','siddhant','dakshina','bikharna','charitra','pahunche','suraksha','paryatan','taiyaari','tatkalin','ghinouna','facebook','protidin','porporle','sheshtai','parvachan','vichchhed','chopadiya','dhaaranaa','baksheesh','sangharsh','sanrachna','vyavastha','nishpatti','chikitsak','sindhutva','dhakelana','giraftaar','jolkhabar','dhanyavaad','niyantraan','pratishodh','swatantrata','pratiyogita','bondhuchara','pratispardha'],
    'Telugu': ['mee','adi','ani','idi','ela','oka','emi','naa','tho','adu','ala','baa','edo','haa','ila','jey','ooh','ore','nenu','kuda','kani','idhi','inka','vala','ante','adhe','okka','aame','adhi','anta','arey','ayyo','levu','leka','sepu','tosi','aaga','aena','aina','aite','amma','atla','ayya','eyyi','gari','hari','inni','itla','jaya','kala','keka','kodi','mari','menu','memu','raja','sari','seva','tanu','kosam','kooda','manam','avunu','aithe','ledhu','tappa','vaadu','kotha','kante','vaadi','ninnu','emito','pedha','kadaa','nannu','adugu','baaga','cheri','daani','desam','dhani','intlo','meeku','meeru','nijam','nundi','okati','oorlo','paalu','paata','pilla','prema','sagam','saavu','seema','sodhi','sompu','tunne','abbay','anthe','asalu','bandi','bhalu','chesi','chota','frnds','gaali','goppa','ipudu','jeyya','kayya','lokam','okaru','osaru','pedda','randi','satya','sarle','srinu','thodu','tholi','vachi','valla','yenti','yokka','unnadu','unnaru','antaru','enduku','avarku','avanni','assalu','baadha','dagara','ichina','illalu','intiki','jarige','kaadhu','kaalam','kastha','manasu','mundhu','panulu','raadhu','rojuki','tosina','vaalla','aasalu','andaru','appudu','bagane','badulu','bayata','bhayya','bhoomi','cheyya','chinna','cinema','dhanni','eyyaru','eyyava','gelavu','guruvu','kavali','lopala','madham','modati','mohini','nenuve','perugu','thindi','vandha','vasthe','cheyali','andamga','kakunda','tappaka','kothaga','matrame','untaadu','istharu','chesina','peddaga','abaddam','maamulu','thakuva','vaadini','padithe','padandi','aasaalu','adugunu','chotuga','dengina','dengali','doshamu','endhuku','evariki','kathalu','kevalam','kshanam','maarina','nijamga','praanam','prajalu','rakanga','rakunda','saraina','sontham','vundali','adugulu','aduthey','dhayyam','gaalilu','gattiga','krishna','madyalo','nenunte','pillalu','rambabu','tarvata','lallike','nainaki','urevalo','mimmalni','avakasam','vachindi','kalisina','cheppanu','anukunta','cheyadam','veskondi','aadarana','avasaram','bhootulu','chudandi','daggarai','erripuka','manchiga','okkasari','paatedhi','padipoya','penchaga','pothunna','prakhyam','prakrame','prayanam','saradaga','sarvasva','vaallaki','vadalara','vishayam','dikhlana','prayatna','shikayat','vyavahar','shradhha','kartavya','siddhant','dakshina','bikharna','charitra','pahunche','suraksha','paryatan','taiyaari','tatkalin','ghinouna','chebuthe','cheyyaru','dhaaniki','jeyyaaru','nenulaki','peddamma','thakkuva','facebook','protidin','porporle','sheshtai','parigetti','vasthundi','chesinatu','avvakasam','kavalsina','raasindhi','antunnaru','cheyyandi','adigindhi','antunnadu','istharani','bauntundi','chinnappa','daridrapu','jeevitham','jolliestu','kalavatha','padutunna','palukutho','prakharam','preminchu','sakshanga','simhiyalu','vichitram','parvachan','vichchhed','chopadiya','dhaaranaa','baksheesh','sangharsh','sanrachna','vyavastha','nishpatti','chikitsak','sindhutva','dhakelana','giraftaar','jabardast','jarigindi','meerulaki','jolkhabar','theeskondi','isthunnaru','adugutundi','isthunnanu','brathakali','chesthunna','kaaranamga','prushottam','regincharu','sandarbham','dhanyavaad','niyantraan','pratishodh','isthunnaadu','vasthunnaru','telusthundi','aaparaadham','dorakatledu','jeevithamlo','marichipoya','paatinundhi','paristhithi','swatantrata','pratiyogita','gelusthunna','lekhinchuko','srimanthudu','thoguthundi','bondhuchara','chesthunnaru','kanipettandi','chepthunnadu','bhayapettina','emaipothunna','jaruguthundi','kanipincheru','modatinunchi','pratispardha','chesthunnaadu','anukuntunnanu','maatladuthunu','sambandhinche','choosthunnaadu','aalochinchandi'],
    'Tamil': ['nan','ida','kai','vaa','kal','kol','kan','poy','men','mun','oru','sav','sol','svk','idu','por','pul','vil','aal','maa','nee','yen','avan','aval','illa','athu','podi','peru','vaai','vidu','seer','vitu','meel','ulla','mara','pada','aana','aaha','adhu','anbu','chol','eppo','etho','inge','ippa','ival','ivar','kaal','kana','koor','keer','naan','neer','nool','onru','osai','oyil','paal','paar','pasi','savu','seri','epdi','ithu','kann','koll','maan','meip','puvi','ravu','soll','than','thol','maip','aadu','aatu','avar','ayya','enna','enru','kelu','kodi','kudi','laam','siru','veru','intha','alavu','nalla','sollu','kooda','veesu','pottu','solla','aasai','nilai','porul','solli','aanal','avaru','boomi','engal','enjoy','indha','jolly','kalvi','kanda','kaval','kadal','koduk','kurai','maari','mahan','magan','manam','mella','mozhi','naadu','nalam','ninga','padal','padam','pagal','pothu','pudhu','raaja','ruchi','saara','sadai','samam','selai','surya','tamil','tarum','thaan','thala','endru','engum','ethai','payir','peyar','saami','sanda','there','illai','keezh','kuyil','pokku','ponnu','rasam','velai','vetti','aatam','avala','ennai','innum','kelvi','kovil','meeru','mokka','namma','naanu','neevu','paaru','summa','ungal','unmai','unnai','venum','yethu','neeyum','thaane','eppadi','aanaal','anuppu','thayir','unakku','enakku','suzhal','veettu','piragu','pakkam','selavu','thothu','umakku','vaikku','agalam','baasha','badhil','chithi','ippadi','ivarai','jeevan','kanavu','keerai','kollai','iyalbu','kangal','makkal','mazhai','moolai','mudhal','nanban','nandri','nangal','needhi','nirkka','parisu','poonga','raatri','sandai','thanni','kaalai','karuvi','kilavi','parvai','poonai','sakthi','seemai','selvam','thatha','ratham','thanga','tharum','thedum','irukku','iruvar','kaattu','kathai','kathir','konjam','maanam','maattu','neenga','oruvar','paavam','periya','panniru','thavira','irukkum','migavum','kevalam','vilakke','veliyil','petraar','poorvam','vayathu','vilakka','pattaam','athigam','amaippu','avanuga','azhudhu','ethuvum','ippavum','iyakkam','kadhali','kanneer','kavalai','kodutha','irunthu','karuthu','manaivi','marakka','munneer','odhunga','paartha','paarvai','payanam','sooriya','sundari','thangam','kadalai','kadavul','kurippu','magimai','manidha','maranam','rasathi','sappadu','thanjam','kodumai','puthusu','senthil','thanjai','avargal','enpathu','irukken','iruppin','ithuvum','mudiyum','naankal','nammaku','samayal','samayam','solriya','thamizh','unpathu','valathu','illaamal','tharuvom','illaatha','thiruppi','mukkiyam','kudumbam','parandhu','thiruthi','pannalam','purindhu','aruginra','pannunga','kalakkal','kavingar','kidaikka','ivarukku','manathil','mannavan','marundhu','puthumai','tharunam','ivalavil','kannamma','puthagam','thirudan','irupathu','kulithal','sandroru','thodarbu','yosithan','aarambam','avudaiya','kozhambu','marupadi','munnaadi','naanukku','sollunga','solvathu','tholaikka','aarambham','kaalathai','madhiyaal','nannaivar','sandhippu','thagappan','mazhaiyil','ragasiyam','kanavugal','magizhchi','avarkalai','engalukku','irunkiren','naanungal','periyavar','ungalukku','paravaigal','bhagavatha','kuzhandhai','olarvaatha','paarkalaam','makizhndhu','ratchasiya','tharavidai','vilaiyattu','azhaikanum','neengaluku','sugathiram','irukkirathu','padikkiraan','kudikkiraan','kottindraar','kodukkiraar','kodukkiraan','aarambhikka','nadanthathu','nedunthadhu','rajinikanth','marupadiyum','pudhupettai','neengalukku','puriyavillai','anaivarukkum','sooriyanaaru','yethentruyil','solvadhillai','kandupidippom','sagodharargal','virumbukirathu','kaattirukkirathu','koduthirukkiraan','maranthirukkirathu'],
    'Marathi': ['ahe','ani','kal','nay','dil','kay','aai','aaj','aas','bag','dar','dev','dur','has','jag','jau','kha','lat','mol','vel','sap','sut','zep','mala','kasa','sang','kaay','kase','asla','khar','pani','dili','aala','nahi','kela','tula','gheu','yete','raha','asli','kaam','kahi','kele','karu','aho','ala','ali','ari','asa','asi','ata','aani','aata','amhi','ahaa','amba','amha','anek','baba','bahu','bala','bhat','bhas','chal','dada','fakt','gela','ghar','ghon','haat','hasu','hona','hoti','jaau','jaga','jeve','jhal','jati','kaal','keli','khan','khup','lage','lagn','lakh','maan','mann','mast','maza','mazi','nako','vaat','vish','puja','roka','sant','sarv','thav','ubha','saath','aaple','kaahi','sagla','majhe','kuthe','tyala','bagha','sagle','sangu','disat','ajab','akal','alag','amaa','amar','anga','anya','apla','aple','apli','apun','asud','aamhi','aapan','accha','agadi','ajuba','anand','ajaba','aatak','aakul','aanek','aarth','adhik','badal','baher','bahin','bhaag','bhaav','chhan','chaar','darja','dekhu','divas','dolya','durga','fayda','fokat','gaaon','gosht','gotra','jatee','jhali','kadha','kadun','kalat','kamal','karun','khara','maaji','maane','madat','majha','naahi','naate','navin','vadal','vakta','vhaag','vilas','chowk','latur','punha','paisa','prera','punah','punar','sapna','sathi','sathe','savin','thamb','thaya','upyog','vrudh','zepun','majhya','jhaale','shakto','milale','shakti','mhatla','aaplya','saathi','kuthhe','shikli','milala','tyacha','aaval','agale','aikya','ajali','ajata','amule','anang','anant','angat','anjay','artha','asach','ashat','asita','asudh','ayush','adhika','aamcha','aadhar','aagman','aanand','amucha','arthik','adbhut','aghadi','acchaa','bagait','balaka','bhaaji','bhojan','chaalo','dolyaa','geleli','ghevun','jeevan','keleli','laagal','lakhun','lavkar','mannat','mhanun','nantar','nantre','vachan','vaidya','vishay','vividh','mhanje','prayog','pushpa','rakhun','rustom','sanman','sathev','swagat','tvacha','tumhala','prashna','kuthlya','aamhala','rahasya','kaarane','amchya','anasha','anjali','anupam','apunle','arogya','asleel','asmita','asunuk','atavar','athava','athvan','abhyaas','amuchya','aakshar','aarthik','aananda','aadarsh','aabhaar','barobar','bhajani','bhraman','chukoon','darshan','hijaade','kashala','maulana','vinanti','waastaw','prerana','saangun','sahitya','sampati','sweekar','swataha','thambun','vaachan','watpade','sangitla','jhalayla','sangitli','shikshan','adarsha','adnyaat','alaukik','angikar','anubhav','anukram','anusara','anyavar','apeksha','apharan','aphilan','athavan','ayushya','aamuchya','abhimaan','abhipray','gadhadya','gandhiji','gharchya','jagachya','khushaal','lahanpan','wavering','randhawa','sangitale','dakhavata','alankaar','anarghya','anubhava','anukampa','anusaran','apekshit','aradhana','asankhya','aakrandan','aashirwad','aakrandit','abhipraya','bhavishya','karnyacha','mumbaichi','sangharsh','swatantra','vatavaran','apunyapas','asachahe','asudhahe','bhrunhatya','gadhvachya','instrument','mumbaichya','anyavastha','asudhasel','atishuddha','abhyaasacha','anusarkeli','asudhaslya','asunukasel','angikarkeli','arthashastra','asunukaslya','angikarkarun'],
    'Urdu': ['mai','aap','hai','kya','yeh','par','kar','iss','aur','jis','bhi','tum','dil','sab','koi','kam','hun','rha','rhi','aag','aah','hum','log','maa','nah','umr','uss','woh','aib','nau','tha','aaj','asi','ata','ati','aye','bai','but','dar','din','dum','mein','hain','kiya','hota','agar','kaam','kuch','kyun','dard','wakt','acha','baar','sath','kisi','apna','bana','uska','unka','jana','phir','aana','apne','usne','unki','haan','pari','meri','mujh','raha','rahi','aage','aate','aati','aaya','aaye','adaa','faiz','haal','khul','laal','lafz','lage','lahu','liye','maan','maut','mere','munh','naam','nahi','paas','raah','raaz','rooh','saba','sada','soch','wafa','alag','ansu','asar','badi','chup','dafn','date','fana','fikr','gair','gham','ghar','gila','hala','ishq','jaan','jama','kash','laut','lime','lutf','maat','ruju','saja','shak','suna','zaat','adab','chot','daam','deta','husn','jurm','khat','maah','maal','aisa','aisi','ajab','alas','aman','ankh','bala','beta','bich','bura','daag','dagh','dukh','duri','yahan','kuchh','kaise','mujhe','dunya','tarah','dusra','karna','larka','larki','tumhe','taqat','sakta','sakti','maine','aapas','faqat','fikar','haath','habli','hafiz','havaa','khush','lagta','lekin','milti','naqsh','pahle','pehle','rooth','sapna','shauq','subah','udhar','umeed','waada','aankh','afsos','ajeeb','aksar','alfaz','ambar','aqsar','araam','azaad','bahar','bahut','chand','dilon','ehsas','hadsa','irada','jahan','judai','karam','khwab','laund','nahin','naseb','nasib','sabab','sahib','sajde','shair','sunte','surat','udasi','ujala','zeest','zuban','afwah','anban','arman','aurat','baita','dafan','daman','dinon','diqat','firqa','garaj','gusht','irade','jaaiz','kalma','khauf','khayr','likha','aapna','achha','adaab','aisay','akela','akhir','ameer','anmol','asman','asrar','ateet','atish','awaaz','bacha','badla','badle','bahot','behad','belam','betab','bijli','burai','burna','dagha','dekha','dosti','bilkul','zaroor','aakhir','aaraam','ghazal','haafiz','haasil','hadees','halaat','haseen','khwaab','maanta','maarna','nafrat','naseeb','piyaar','qudrat','tanhaa','afwaah','akhiri','baithe','bayaan','bedaad','doosri','faasla','haazir','haveli','iltija','inteha','khabar','khushi','khyaal','maamla','mayoos','munsif','mutaal','napaak','qusoor','siyaah','sunkar','taaluq','thakan','tufaan','ummeed','aasman','aayaat','afsona','aftaab','alfaaz','almari','aqwaal','aziyat','bahana','bhookh','dulhan','duniya','ehsaas','hazaar','hijaab','imkaan','ilaahi','insaan','jalebi','jawaab','khabri','mareez','masail','mehfil','moorat','muflis','mutalq','aapnay','adhuri','ajnabi','alvida','ankhon','anjaan','anjali','asmani','astana','bai-ji','bejaan','bemari','benaam','beshak','beshaq','bewafa','bidaar','bidesi','bilakh','chehra','chhota','chhoti','chumma','dekhna','dekhti','dekhte','doston','hairaan','mehmaan','musafir','sunehri','talaash','tehzeeb','zamaana','zindagi','amaanat','amreeka','aurtain','bewafai','faryaad','haaziri','ijaazat','mojaiza','munasib','mushkil','musibat','raushan','riwayat','safeena','shaukat','tasweer','aanchal','adaalat','afwaahe','akhbari','ambwari','anjuman','baghair','chandni','ijtimaa','irshaad','masroof','mehkuma','munafiq','aitbaar','badnaam','badnami','barsaat','bekarar','bekhauf','bekhudi','bemisal','beqarar','burhapa','burhapy','charagh','chashma','chubhti','daikhna','darogha','dilruba','aazmaish','mehrbani','mulaqaat','nazaakat','paimaane','sharafat','behisaab','imtihaan','mohabbat','naqaabil','pakeezah','pareshan','samandar','tamaasha','tanaffur','tashreef','baahisht','charaagh','intezaar','awarapan','bahut-sa','bahot hi','bardasht','bekhabar','beniyaad','beparwah','darkhast','darwaaza','khwaahish','maashooqa','nigaahein','intikhaab','khairiyat','badmaashi','hoshyaari','istiqlaal','khazaanaa','asaathiya','badi waqt','beikhtiar','chhoti si','darmiyaan','dekha hua','dilrubaai','ashiyaanaa','aisi tarah','bahut zada','but-parast','dard-e-dil','dekhti hui','dil-e-wafa','dil-o-jaan','dukh-e-dil','asar-e-wafa','ata-ul-faiz','awaaz-e-dil','chhoti umar','dafa-e-ishq','dard-e-ishq','dard-e-wafa','dil-e-nazuk','dukh-e-wafa','badi mushkil','bahut zaruri','chhoti ulfat','dard-e-sitam','dukh-e-jahan','bahut zaroori','bilakh-bilakh','dekhte-dekhte','dekhte-dekhti','asar-e-mehboob','bahut pyaar se','dekhte-dekhtay','darkhast-e-ishq','dil-e-nazuk-e-ishq'],
    'Gujarati': ['ema','kya','che','ane','kem','aap','hoi','thi','kai','maj','shu','cho','koi','laj','nai','nav','oli','evu','naj','nik','por','pun','roj','ame','are','ave','bag','bol','evi','tel','des','han','hun','mel','vat','vay','kaj','mul','sau','tame','chhu','etla','kari','chho','rahe','vaat','hova','natu','maal','karu','hase','chhe','game','hoyo','kaho','kare','raha','haru','bura','besu','choo','jovu','kaya','kevi','loko','mari','masu','navu','puja','raas','rite','ruso','thay','toon','vato','janu','joya','kajo','karo','lakh','laav','lage','maja','mast','moti','motu','naam','phal','pote','pray','rato','shak','sukh','vadh','vish','ajab','amne','biji','etle','javu','lovu','rath','sath','seva','shri','vari','vaid','vhal','bhir','dhan','divs','evan','halt','jova','kone','mane','nava','path','same','sane','soni','tale','varg','agad','ekta','faje','feni','gidh','javi','koni','loha','mate','mine','mota','nahi','nath','vayu','maate','shako','bhale','ghana','ramat','thase','koine','sharu','badhu','bhaji','bolvu','dabbo','dheku','jamva','kadhi','karvu','kashu','ketlu','kharu','laadu','mathi','methi','mithu','nathi','phodi','pishi','saaru','Shano','swaad','upmaa','vagar','vandu','vilay','aamka','aapvu','aathu','aavso','janya','kahvu','lakhu','lahan','lapet','palav','pujar','puvar','ratri','vahen','badha','bagay','darni','divas','jaher','jarur','jyare','kadak','kahee','kevay','laher','nathu','rojnu','svadu','vagad','avtar','bhedu','botad','chana','desma','dikri','jagma','khena','nakko','nirav','pauna','pehlu','rashi','ratan','rotlo','sovak','vikas','amare','desni','faraj','halol','india','jaman','kamna','lasan','lokta','melaa','naram','palan','rasoi','ratva','sompu','vadhu','tamara','khabar','sharab','bakadu','kariye','naathi','paneer','pranam','vachan','varsad','abhaag','ishwar','karine','khichu','maarnu','nishan','rajkot','salaah','swaadu','tempik','vanday','vilamb','aadhik','aaviye','apnavu','asarvu','damani','dhvani','dhokla','nimitt','nirnay','oldage','padavo','pakshi','sarkar','wisdom','abhyas','agrahy','bhaili','bhurak','chodvu','dagmag','kutchh','mariyu','pragat','shriji','vichar','akhand','bagdum','dhokra','divase','jagran','sanket','shakha','soorna','trance','vishay','bharela','chokari','dhandho','khandvi','khichdi','pragati','vadhare','vargani','vartavo','wartime','kharaab','kharedi','mysooru','navsari','shikhar','thaatha','vandana','vartman','vichaar','vitaran','vrushti','aayojit','ishware','mathama','pehchan','shaamak','vadhana','vartalo','village','adhunik','devgadh','hanuman','panchma','aradhya','baharva','itihaas','marvaad','mulakat','nishtha','sukhkar','vicharo','vikasak','bhakhari','dandvatu','nishaani','samajhvu','varshaam','aksharvo','khichadi','vicharan','dhandhal','suvichar','vartaman','villayat','avinashi','prayogik','varganim','bhavishya','ghamadhna','shyamahoo','ladakvaya','mazamaaze','vicharano','anandkand','parikrama','savaarish','sukhakari','salaamati','punyannand','icchapurti','punyabhumi','dhandhalya','nirikshana','prernapurna','khushamadee','punyabhoomi','shubharambh','randhikkaran'],
    'Kannada': ['idu','adu','ene','illa','mele','eega','nivu','hege','beku','hosa','yenu','yava','ella','naanu','nimma','yaava','yaake','neenu','avaru','nimge','maadi','tumba','haage','enadu','yella','haagu','neeve','yaaru','namma','neevu','saaku','naavu','aagide','namage','ellide','ellavu','madhya','madhye','barutte','anisutte','maadiddiya'],
    'Malayalam': ['ini','nee','oru','pal','kai','njan','illa','aara','avan','athu','enne','ende','ente','aaru','undu','aanu','avar','entha','enthu','aranu','venam','athil','koodi','ningal','thanne','ingane','kaanam','aarude','karuthi','sahayam','cheiythu','koduthal','cheyyuka','enthelum','kudumbam','prashnam','pattonnu','ningalude','arikkilla','irikkanam','santhosham','aayirikkanam','kandupidichu','samsarikkunnu','paranjirunnilla','cheiythirunnilla'],
    'Punjabi': ['eta','ahe','eti','hai','pai','hei','att','cha','dil','fer','hun','jee','nhi','par','vai','bas','hoi','aan','nai','eho','hor','rab','deo','ice','ujj','tume','kaun','heba','kari','kahi','agge','assi','bhut','boli','ghar','haan','hass','hoya','jatt','kadd','khad','kujh','kuri','mann','menu','mera','meri','nahi','vali','vich','teri','wand','chaa','dass','daru','gedi','rabb','ankh','door','ishq','jeha','boot','hoye','paro','brar','dayi','kamb','patha','bhala','etiki','bhalo','balle','chaan','cheez','dhyan','ditta','fikar','gabru','haasa','kammi','kardi','khani','kinni','kitta','laggi','laina','lambi','mainu','majaa','mithi','vadde','saanu','thodi','turdi','janda','haigi','dassi','hunda','bulli','daaru','disdi','sajna','akhan','hoigi','kinna','paake','vekhi','bacha','billo','chete','chhad','hassi','lagge','maape','hunde','boldi','chhan','dekho','heavy','karan','lutti','paiye','vaari','bhabi','dasso','dukhi','gaana','kemiti','hauchi','hebaku','tumaku','parilu','aayaan','dekhdi','ghumdi','hassdi','khaana','luteya','nakhra','punjab','vekhna','tuhada','painde','changa','brinda','channa','mainnu','tuhanu','bhutta','changi','jeonde','kacche','khushi','aashiq','bhangra','charhda','chhutti','balliye','teriyan','punjabi','valiyan','vanjhali','vaddeyan','mutiyaran'],
    'Odia': ['aau','ama','hei','jau','asi','hau','jiu','oka','aru','asa','odi','ori','naa','sei','aap','abe','aha','aja','ala','api','ari','aum','bapa','asti','boli','asta','ithi','pain','sabu','tame','tora','aaji','anna','bapu','bati','khia','loka','mane','jiba','mote','odia','thae','aama','hela','siba','nahi','suna','aaja','aala','aame','abhi','alag','alai','amar','anya','apye','atma','bhala','chhai','odhia','chali','poila','naahi','bhalo','sathi','thili','amaku','chhua','dusta','thata','amara','artha','asiba','jiban','aapna','achha','adugu','ahare','ajeya','alada','amate','anand','aneka','anila','arati','asana','chhata','jeeban','paithi','sasthi','bandha','lagila','asuchi','bhaaji','kanhei','rahilu','bhanga','hauchi','karibe','thatha','rauchi','aitare','alaita','alpana','amruta','ananta','ananya','anesha','aniket','animan','anyata','apiuna','apurna','aputra','arogya','asatya','asmita','asurya','avidya','dekhilu','karuchi','bihanga','dekhela','rakhiba','boluchi','chadhei','rahuchi','adbhuta','alaakhi','alahasa','alaukik','alokika','aniyata','anubhav','anusara','anupama','ashubha','asuddha','astitva','aumkara','avastha','avinita','avirati','avyakta','alaghana','alochana','aneshana','anindita','aniruddh','anyatara','apavarga','aradhana','atmatapa','aupasana','annapurna','jaukanthi','anukarana','anusarana','apaharana','aparajita','apasiddha','atmajnana','atmavidya','aupadhika','anantatapa','anekanetra','aneshapana','apasavadhi','asmitatapa','asuravrata','atmavritti','anantajnana','anilashakti','anupamaguni','atmanishtha','avasthapana','aviratatapa','avyaktatapa','animanishtha','apurnashakti','asmitavritti','aniruddhatapa','aviratavritti','avyaktavritti','anubhavashakti','aniruddhashakti','aniruddhavritti']
}

def Custom_language_detection(ytcomment):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function uses custom words from regional languages and find exact roman script of indian languages
    :param ytcomment: data frame
    :type ytcomment: dataframe
    :return: dataframe
    """
    for language, words in words_to_check.items():
        for word in words:
            if word in ytcomment['comment_textDisplay']:
                ytcomment['language'] = language
                return ytcomment
    return ytcomment
###############################################################################################
#Step14: Additional iteration in updating the language code of Indian regional languages based on step13
#Languages list which are considered for analysis
Language_lookup = {
    'language': ['English', 'Hindi', 'Bengali', 'Telugu', 'Tamil', 'Marathi', 'Urdu', 'Gujarati', 'Kannada', 'Malayalam', 'Punjabi', 'Odia'],
    'lang_code': ['en', 'hi', 'bn', 'te', 'ta', 'mr', 'ur', 'gu', 'kn', 'ml', 'pa', 'or']
}
Language_lookup = pd.DataFrame(Language_lookup)
def Custom_language_code_mapping(ytcomment):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function updates the language code based on revised language
    :param ytcomment: data frame
    :type ytcomment: data frame
    :return: data frame
    """
    language = ytcomment['language']
    if language in Language_lookup['language'].tolist():
        lang_code = Language_lookup[Language_lookup['language'] == language]['lang_code'].values[0]
        ytcomment['language_code'] = lang_code
    return ytcomment
################################################################################################
#Step15: convert english based comments to lowercase
def English_comments_to_lower(sourcedata, columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function converts english comments to lower case
    :param sourcedata: The source data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :return: data frame
    """
    sourcedata[columnname] = sourcedata[columnname].str.lower()
    return sourcedata
################################################################################################
#Step16: Identify the stop words and remove
def Stopwords_detection_removal(columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes the stop words from comments
    :param columnname: The comments text column
    :type columnname: str
    :return: str
    """
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = stopwords.words('english')
    custom_stopwords = ['fraud', 'thevidiya', 'otha', 'pagal']
    textprocessed = columnname
    textprocessed.replace('[^\w\s]', '')
    textprocessed = " ".join(word for word in textprocessed.split() if word not in stop_words)
    textprocessed = " ".join(word for word in textprocessed.split() if word not in custom_stopwords)
    textprocessed = " ".join(Word(word).lemmatize() for word in textprocessed.split())
    return(textprocessed)
################################################################################################
#Step17: Source didnt had label. We are adding labels based on tags
def CreateFlagsbyLabelingParty(sourcedata):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function creates 2 flag columns to identify the comments is based on BJP or Congress
    :param sourcedata: data frame
    :type sourcedata: data frame
    :return: data frame
    """
    bjp_keywords = [
        'bjp', 'rss', 'modi', 'nda', 'aiadmk', 'pmk', 'bjp', 'dmdk', 'tmc', 'shs', 'jd', 'akali', 'jaya', 'panneerselvam', 'mgr', 'ramadoss', 'vijayakanth', 'paneer', 'bharatiya janata party', 'shiv sena','all india anna dravida munnetra kazhagam', 'janata dal', 'shiromani akali dal', 'pattali makkal katchi', 'lok janshakti party', 'desiya murpokku dravida kazhagam', 'bharath dharma jana sena', 'asom gana parishad', 'apna dal', 'puthiya tamilagam', 'puthiya needhi katchi', 'tamil maanila congress', 'all jharkhand students union', 'bodoland people', 'nationalist democratic progressive party','kerala congress', 'rashtriya loktantrik party','all india n.r.congress','sumalatha', 'right wing', 'religious', 'hindu', 'namo', 'sarkar', 'jagan','thamarai', 'chokidar', 'chowkidaar', 'yogi', 'communal', 'sree ram', 'ram', 'shri', 'rama', 'bharat mata ki', 'siya', 'sri', 'siri', 'guru', 'bhakt', 'mata', 'b j p', 'bhartiya', 'bajrang', 'amit', 'sita', 'lord', 'owaisi', 'baba', 'krishna', 'modhi', 'mulayam',
        'பிஜேபி', 'அகில இந்திய அண்ணா திராவிட முன்னேற்றக் கழகம்', 'அசோம் கண பரிஷத்', 'அப்னா', 'அனைத்து ஜார்க்கண்ட் மாணவர் சங்கம்', 'ஆர்எஸ்எஸ்', 'இராமர்', 'எம்ஜிஆர்', 'கிருஷ்ணா', 'கேரளா காங்கிரஸ்', 'கோமான்', 'சமயம்', 'சர்க்கார்', 'சவ்கிடார்', 'சியா', 'சிரி', 'சிவசேனா', 'சீதை', 'சுமலதா','சோகிதர்', 'டிஎம்சி', 'தமிழ் மானிலா காங்கிரஸ்', 'தேசியவாத ஜனநாயக முற்போக்குக் கட்சி', 'பக்த்','பட்டாலி மக்கள் கட்சி', 'பஜ்ரங்', 'பாபா', 'பாரதிய ஜனதா கட்சி', 'பாரதியா', 'பாரத் தர்ம ஜன சேனா','பாரத் மாதா கி', 'பாஜக', 'போடோலாந்து மக்கள்', 'மாதா', 'முலாயம்', 'மோடி', 'மோதி', 'யோகி', 'ராமதாஸ்', 'ராஷ்டிரிய லோக்தந்த்ரிக் கட்சி', 'லோக் ஜனசக்தி கட்சி', 'வகுப்புவாதம்', 'வலது சாரி', 'விஜயகாந்த்', 'ஜனதா','ஜெகன்', 'ஜெயா', 'ஸ்ரீ', 'ஸ்ரீ ராம்', 'ஷிரோமணி அகாலி', 'அதிமுக', 'அமித்', 'இந்து', 'குரு', 'தாமரை', 'தேசிய முற்போக்கு திராவிட கழகம்', 'தேமுதிக', 'நமோ', 'பன்னீர்', 'பன்னீர்செல்வம்', 'பாமக', 'பிஜேபி', 'புதிய தமிழகம்','புதிய நீதி கட்சி', 'ராமர்', 'ஸ்ரீ',
        'అకాలీ','ఆల్ జార్ఖండ్ స్టూడెంట్స్ యూనియన్', 'అమిత్', 'అప్నా పప్పు', 'అశోమ్ గణ పరిషత్','బి జె పి', 'బాబా', 'బజరంగ్', 'భక్తి', 'భారత్ మాతా కీ జై', 'భరత్ ధర్మ జనసేన', 'భారతీయ జనతా పార్టీ', 'భారతి', 'బోడోలాండ్ ప్రజలు','చోకిదార్', 'చౌకిదార్', 'మతపరమైన', 'గురువు', 'హిందూ', 'మైక్రోసాఫ్ట్ ఎక్సెల్', 'జనసేన', 'జనతా దాల్', 'కేరళ కాంగ్రెస్', 'శ్రీ కృష్ణుడు', 'లోక్ జనశక్తి పార్టీ', 'ప్రభువు', 'మాతా', 'ఎంజిఆర్', 'మోడీ', 'మోదీ.', 'ములాయం', 'నమో', 'జాతీయవాద ప్రజాస్వామ్య ప్రగతిశీల పార్టీ', 'ఒవైసీ','పనీర్ అర్ధం తెలుగులో', 'పన్నీర్ సెల్వం', 'పొట్టేలు', 'రాముడు', 'రామదాస్', 'రాష్ట్రీయ లోక్ తాంత్రిక్ పార్టీ', 'మతపరమైన', 'కుడి వింగ్','ఋశ్శ్', 'సర్కార్', 'శిరోమణి అకాలీ దాల్', 'శివసేన', 'శ్రీ', 'సిరి', 'సీత', 'శ్రీ రామ్', 'మా తండ్రిగారైనా', 'సుమలత', 'తమిళ మాణిక్యాల కాంగ్రెస్', 'తమరై', 'టిఎంసి', 'విజయకాంత్', 'యోగి', 'పవన్ కళ్యాణ్', 'జెఎస్పి', 'కమలం',
        'भक्त', 'बी जे पी', 'सरकार', 'अकाली', 'अपना दाल', 'अमित', 'असम गण परिषद',  'आरएसएस', 'आल झारखंड स्टूडेंट्स यूनियन', 'एन डी ए', 'कला एकीकृत परियोजना', 'गुरु का स्टर्लिंग', 'चोकीदार', 'चौकीदार', 'जगन', 'जन सेना', 'जया', 'जाति', 'जेएसपी', 'टीएमसी', 'डेमोक्रेटिक प्रोग्रेसिव पार्टी', 'तमिल मनिला कांग्रेस', 'थामराई', 'दाहिना विंग', 'धार्मिक', 'नमो', 'पन्नीरसेल्वम', 'पवन कल्याण', 'बजरंग', 'बीजू जनता दल', 'बीजेपी।', 'बोडोलैंड के लोग', 'भरत धर्म जनसेना', 'भारत माता की जय', 'भारतीय', 'भारतीय जनता पार्टी', 'मुलायम', 'मोदी', 'मोधी', 'योगी', 'रामदास', 'राष्ट्रीय लोकतांत्रिक पार्टी', 'रैम', 'लोक जनशक्ति पार्टी', 'शिरोमणि अकाली दाल', 'शिव सेना', 'श्री', 'श्री', 'श्री राम', 'श्रीराम', 'सिया', 'सिरी', 'सीता', 'स्वामी', 'हिंदू',
        'আকালি', 'অল ঝাড়খণ্ড স্টুডেন্টস ইউনিয়ন', 'অমিত', 'বজরং', 'ভক্ত', 'ভারত মাতা কি', 'ভারত ধর্ম জন সেন','ভারতীয় জনতা পার্টি', 'ভারতিয়া', 'বিজেপি', 'বোডোল্যান্ডের মানুষ', 'চকিদার', 'চৌকিদার', 'সাম্প্রদায়িক', 'গুরু', 'হিন্দু', 'জন সেনা', 'জনতা ডাল', 'কৃষ্ণা', 'প্রভু', 'মাতা', 'মোডী', 'মোডী', 'মুলায়ম', 'নামো', 'এনডিএ', 'রাম', 'রামা', 'রাষ্ট্রীয় লোকতান্ত্রিক পার্টি', 'ধর্ম', 'ডান পাখা', 'আরএসএস', 'সরকার','শিরোমণি আকালি ডাল', 'শিবসেনা', 'চিঠি বন্ধুকে লেখা', 'সীতা', 'সিয়া', 'শ্রী রাম', 'শ্রী', 'ফুল', 'টিএমসি', 'যোগী',
        'अकली', 'एएमआयटी', 'बाबा', 'बजरंग', 'भक्त', 'भारत माता की', 'भारथ धर्म जन सेना', 'भारतीय जनता पार्टी', 'भारतीया', 'बीजेपी', 'बोडोलॅंड लोक', 'चोकीदार', 'चौकीदार', 'सांप्रदायिक', 'गुरु', 'हिंदू', 'जनसेवा', 'जनता दाल', 'कृष्णा', 'लोक जनशक्ती पार्टी', 'लॉर्ड', 'माता', 'मोधी', 'मोदी', 'मुलायम', 'नमो', 'एनडीए', 'रॅम', 'राम', 'धार्मिक', 'उजव्या पंख', 'आरएसएस', 'सरकार', 'शिवसेना', 'श्री.', 'एसएचएस', 'सिरी', 'सीता', 'सिया', 'श्री राम', 'श्री', 'कमळ', 'योगी',
        'اکالی', 'امیٹ', 'بابا', 'بجرنگ', 'بھکت', 'بھارت ماتا کی', 'بھارت دھرم جنا سینا', 'بھارتیہ جنتا پارٹی', 'بھارتیہ', 'بی جے پی', 'چوکیدار', 'چوکیدار', 'فرقہ وارانہ', 'گرو', 'ہندو', 'جنا سینا', 'جنتا دال', 'کرشنا', 'لارڈ', 'ماتا', 'مودھی', 'مودی', 'ملائم', 'نمو', 'این ڈی اے', 'رام', 'راما', 'مذہبی', 'دائیں بازو', 'انتہائی سادہ سندیکت', 'سرکار', 'شیرومنی اکالی دال', 'شیو سینا', 'شری', 'سیتا', 'سری رام', 'سری', 'لوٹس', 'یوگی',
        'અકાલી', 'એમીટ', 'બાબા', 'બજરંગ', 'ભક્ત', 'ભારત માતા કી', 'ભારત ધર્મ જન સેના', 'ભારતીય જનતા પાર્ટી', 'ભારતીયા', 'બ્જ્પ', 'ચોકીદાર', 'ચોકીદાર', 'સાંપ્રદાયિક', 'ગુરુ', 'હિંદુ', 'જન સેના', 'જનતા દળ', 'કૃષ્ણ', 'સ્વામી', 'જય માતા દી', 'મોઢી', 'મોદી', 'મુલાયમ', 'નમો', 'એનડીએ', 'રેમ', 'રામ', 'ધાર્મિક કટ્ટરવાદ', 'જમણી પાંખ', 'આરએસએસ', 'સરકાર', 'શિવ સેના', 'શ્રી', 'સીતા', 'શ્રી રામ', 'શ્રી', 'કમળ', 'તમક', 'યોગી',
        'ಅಮಿತ್', 'ಬಾಬಾ', 'ಭಜರಂಗ್', 'ಭಕ್ತಿ', 'ಭಾರತ್ ಮಾತಾ ಕಿ', 'ಭಾರತ್ ಧರ್ಮ ಜನಸೇನಾ', 'ಭಾರತೀಯ ಜನತಾ ಪಕ್ಷ', 'ಭಾರ್ತಿಯಾ', 'ಬಿಜೆಪಿ', 'ಚೋಕಿದಾರ್', 'ಚೌಕಿದಾರ್', 'ಕೋಮು', 'ಗುರು', 'ಹಿಂದೂ', 'ಜನಸೇನಾ', 'ಜನತಾ ದಾಲ್', 'ಕೃಷ್ಣ', 'ಲಾರ್ಡ್', 'ಮಾತಾ', 'ಮೋದಿ', 'ಮೋದಿ', 'ಮುಲಾಯಂ', 'ನಮೋ', 'ನಲ್ಲಿ', 'ರಾಮ', 'ರಾಮ', 'ಧಾರ್ಮಿಕ', 'ಬಲ ವಿಂಗ್', 'ತುಕ್ಕುಗಳು', 'ಸರ್ಕಾರ್', 'ಶಿವಸೇನೆ', 'ಶ್ರೀ', 'ಸೀತಾ', 'ಶ್ರೀ ರಾಮ್', 'ಶ್ರೀ', 'ಕಮಲದ', 'ಟಿಎಂಸಿ', 'ಯೋಗಿ',
        'അക്കളി', 'അമിത്', 'ബാബ', 'ബജ്രംഗ്', 'ഭക്തി', 'ഭാരത് മാതാ കി', 'ഭാരത് ധർമ്മ ജനസേന', 'ഭാരതീയ ജനതാ പാർട്ടി', 'ഭാരതീയ', 'ബി.ജെ.പി', 'ചോക്കിദാർ', 'ചൗക്കിദാർ', 'സാമുദായിക', 'ഗുരു', 'ഹിന്ദു', 'ജനസേന', 'ജനതാ ദാൽ', 'കൃഷ്ണ', 'യജമാനൻ', 'മാതാ', 'മോദി', 'മോദി', 'മുലായം', 'നമോ', 'റാം', 'രാമ', 'മതം', 'വലത് വിംഗ്', 'സർക്കാർ', 'ശിവസേന', 'ശ്രീ.', 'സീത', 'ശ്രീ റാം', 'ശ്രീ.', 'താമര', 'യോഗി',
        'ਅਕਾਲੀ', 'ਅਮਿਤ', 'ਬਾਬਾ', 'ਬਜਰੰਗ', 'ਭਗਤ', 'ਭਾਰਤ ਮਾਤਾ ਕੀ', 'ਭਾਰਥ ਧਰਮ ਜਨ ਸੇਨਾ', 'ਭਾਰਤੀ ਜਨਤਾ ਪਾਰਟੀ', 'ਭਾਰਤੀ', 'ਬੀ.ਜੇ.ਪੀ', 'ਚੌਕੀਦਾਰ', 'ਚੌਕੀਦਾਰ', 'ਫਿਰਕੂ', 'ਗੁਰੂ', 'ਹਿੰਦੂ।', 'ਜਨ ਸੈਨਾ', 'ਜਨਤਾ ਦਾਲ', 'ਕ੍ਰਿਸ਼ਨਾ', 'ਰੱਬ', 'ਮਾਤਾ', 'ਮੋਧੀ', 'ਮੋਡੀ', 'ਮੁਲਾਇਮ', 'ਨਮੋ', 'ਰਾਮ', 'ਰਾਮਾ', 'ਧਰਮ', 'ਸੱਜਾ ਖੰਭ', 'ਆਰਐਸਐਸ', 'ਸਰਕਾਰ', 'ਸ਼ਿਵ ਸੇਨਾ', 'ਸ਼੍ਰੀ', 'ਸੀਤਾ', 'ਸ਼੍ਰੀ ਰਾਮ', 'ਸ਼੍ਰੀ', 'ਕਮਲ', 'ਯੋਗੀ',
        'ଏଲିନା କହିଲ', 'ମଧ୍ୟରେ', 'ଏଲିନା କହିଲ', 'ଏଲିନା କହିଲ', 'ଏଲିନା କହିଲ', 'ଭାରତ ମାତା କି', 'ଭାରତ ଧର୍ମ ଜନସେନା', 'ଭାରତୀୟ ଜନତା ପାର୍ଟି', 'ଏଲିନା କହିଲ', 'ଚୌକିଦାର', 'ସାମ୍ପ୍ରଦାୟିକ', 'ଏଲିନା କହିଲ', 'ହିନ୍ଦୁର', 'ଜନସେନା', 'ଜନତା ଡାଲ', 'ଏଲିନା କହିଲ', 'ଏଲିନା କହିଲ', 'ମୋଡି', 'ଏଲିନା କହିଲ', 'ଏଲିନା କହିଲ', 'ମେଷ', 'ରାମା', 'ଧାର୍ମିକ', 'ଡାହାଣ ପକ୍ଷ', 'ଏଲିନା କହିଲ', 'ଶିବସେନା', 'ଏଲିନା କହିଲ', 'ଶ୍ରୀରାମ', 'ଶ୍ରୀ', 'କହିଲ', 'ଯୋଗୀ'
        ]

    ing_keywords = ['congress', 'gandhi', 'rahul', 'sonia', 'manmohan',  'pappu', 'dravida munnetra kazhagam','rashtriya janata dal','nationalist congress party', 'janata dal','rashtriya lok samta party','jharkhand mukti morcha','communist', 'marxist','hindustani awam morcha','vikassheel insaan party','muslim league', 'jan adhikar party','viduthalai chiruthaigal','jharkhand vikas morcha','swabhimani paksha', 'bahujan vikas aaghadi','leninist','kerala congress', 'socialist','socialist party', 'marumalarchi dravida munnetra kazhagam', 'mdmk', 'nehru', 'kongres', 'tmc', 'didi', 'bhim', 'jai hind', 'hind', 'mamta', 'communist', 'stalin', 'kanimozhi', 'periyar',  'dmk', 'vck',  'pinarai', 'vijayan', 'Mukti', 'morcha', 'Vikassheel', 'swabhimani paksha', 'kongunadu', 'lalu', 'tejashwi', 'janata dal', 'upendra', 'soren', 'yechury',
        'காங்கிரஸ்', 'யுபிஏ', 'காந்தி', 'ராகுல்', 'சோனியா', 'மன்மோகன்', 'பப்பு', 'திராவிட முன்னேற்றக் கழகம்', 'ராஷ்டிரிய ஜனதா', 'தேசியவாத காங்கிரஸ் கட்சி', 'ஜனதா பருப்பு', 'ராஷ்டிரிய லோக் சம்தா கட்சி', 'ஜார்கண்ட் முக்தி மோர்ச்சா', 'கம்யூனிஸ்ட்', 'மார்க்சியம்', 'இந்துஸ்தானி அவாம் மோர்ச்சா', 'விகாஷீல் இன்சான் பார்ட்டி', 'முஸ்லீம் லீக்', 'ஜன் அதிகாரி கட்சி', 'விடுதலை சிறுத்தைகள்', 'ஜார்க்கண்ட் விகாஸ் மோர்ச்சா', 'ஸ்வாபிமணி பக்ஷா', 'பஹுஜன் விகாஸ் ஆகாடி', 'லெனினிஸ்ட்', 'கேரளா காங்கிரஸ்', 'சமவுடமை', 'சோசலிஸ்ட் கட்சி', 'மருமலாச்சி திராவிட முன்னேற்றக் கழகம்', 'ம.தி.மு.க', 'நேரு', 'டிஎம்சி', 'தீடி', 'பீம்', 'ஜெய் ஹிந்த்', 'பின்', 'மம்தா', 'ஸ்டாலின்', 'கனிமொழி', 'பெரியார்', 'தி.மு.க', 'பினராய்', 'விஜயன்', 'வீடுபேறு', 'மோர்ச்சா', 'விகாஷீல்', 'கொங்குநாடு', 'லாலு', 'தேஜஸ்வி', 'ஜனதா', 'உபேந்திரா', 'சோரன்', 'யெச்சூரி',
        'కాంగ్రెసు', 'ఉప', 'మహాత్మా గాంధీ', 'రాహుల్', 'సోనియా', 'మన్మోహన్', 'పప్పు', 'ద్రవిడ మున్నేట్రా కజగం', 'రాష్ట్రీయ జనతా దాల్', 'నేషనలిస్ట్ కాంగ్రెస్ పార్టీ', 'జనతా దాల్', 'రాష్ట్రీయ లోక్ సమతా పార్టీ', 'జార్ఖండ్ ముక్తి మోర్చా', 'కమ్యూనిస్ట్', 'మార్క్సిస్ట్', 'హిందూస్థానీ ఆవం మోర్చా', 'వికాస్ ‌షీల్ ఇన్సాన్ పార్టీ', 'ముస్లిం లీగ్', 'జన అధికారి పార్టీ', 'విదుతలై చిరుతైగల్', 'జార్ఖండ్ వికాస్ మోర్చా', 'స్వాభిమాని పాక్ష', 'బహుజన్ వికాస్ ఆఘాడి', 'లెనినిస్ట్', 'కేరళ కాంగ్రెస్', 'సామ్యవాది', 'సోషలిస్ట్ పార్టీ', 'మురుమలర్చీ ద్రావిడ మున్నేట్రా కజగం', 'నెహ్రూ', 'కొంగ్రెస్', 'టిఎంసి', 'దయ్యం', 'భీమ్', 'జై హింద్', 'వెనుక', 'మమతా', 'స్టాలిన్', 'కనిమొళి', 'పెరియార్', 'పినారై', 'విజయన్', 'ముక్తి', 'మోర్చా', 'వికాస్ షీల్', 'కొంగునాడు', 'లాలు', 'తేజస్వి', 'ఉపేంద్ర', 'సోరెన్', 'యేచూరి',
        'कांग्रेस', 'यूपीए', 'गांधी', 'हिंदी में कमल का फूल जानकारी', 'सोनिया', 'मनमोहन', 'पप्पू', 'द्रविड़ मुन्नेत्र कड़गम', 'राष्ट्रीय जनता दाल', 'राष्ट्रवादी कांग्रेस पार्टी', 'बीजू जनता दल', 'राष्ट्रीय लोक समता पार्टी', 'झारखंड मुक्ति मोर्चा', 'साम्यवाद', 'मार्क्सवादी', 'हिंदुस्तानी अवाम मोर्चा', 'विकाशील इंसान पार्टी', 'मुस्लिम लीग', 'जन अधिकार पार्टी', 'विदुथलाई चिरुथाइगल', 'झारखंड विकास मोर्चा', 'स्वाभिमानी पक्ष', 'बहुजन विकास आघाड़ी', 'लेनिनवादी', 'केरल कांग्रेस', 'समाजवादी', 'सोशलिस्ट पार्टी', 'मरुमलार्ची द्रविड़ मुनेत्र कझगम', 'नेहरू', 'कोंग्रेस', 'टीएमसी', 'दीदी', 'भीम', 'जय हिन्द', 'विश्वास', 'स्टालिन?', 'कनिमोझी', 'पेरियार', 'पिनराई', 'विजयन', 'मुक्ति', 'मोर्चा', 'विकासशील', 'कोंगुनाडु', 'लालू', 'तेजस्वी', 'उपेन्द्र', 'सोरेन', 'येचुरी',
        'কংগ্রেস', 'উপ', 'গান্ধী', 'রাহুল', 'সোনিয়া', 'মনমোহন', 'পাপ্পু', 'রাষ্ট্রীয় জনতা ডাল', 'জাতীয়তাবাদী কংগ্রেস পার্টি', 'জনতা ডাল', 'রাষ্ট্রীয় লোক সমতা পার্টি', 'ঝাড়খণ্ড মুক্তি মোর্চা', 'কমিউনিজম', 'মার্কসবাদী', 'হিন্দুস্তানী আওয়াম মোর্চা', 'বিকশিল ইনসান পার্টি', 'মুসলিম লীগ', 'জন অধিকার পার্টি', 'লিবারেশন চিতাবাঘ', 'ঝাড়খণ্ড বিকাশ মোর্চা', 'আত্মসম্মানিত দল', 'বহুজন বিকাশ আঘাদি', 'লেনিনবাদী', 'কেরালা কংগ্রেস', 'সমাজতান্ত্রিক', 'সমাজতান্ত্রিক দল', 'নেহেরু', 'কোংগ্রেস', 'টিএমসি', 'দিদি', 'ভীম', 'জয় হিন্ড', 'হরিণী', 'মমতা', 'স্ট্যালিন', 'পেরিয়ার', 'পিনারাই', 'বিজয়ন', 'মুক্ত', 'মোর্চা', 'বিকশিল', 'কঙ্গুন্ডু', 'লালু', 'তেজস্বী', 'উপেন্দ্র', 'সোরেন', 'ইয়েচুরি',
        'काँग्रेस', 'उप', 'गांधी', 'रहुल', 'सोनिया', 'मनमोहन', 'पप्पू', 'द्रविड मुनेत्र काझगम', 'राष्ट्रीय जनता दल', 'राष्ट्रवादी काँग्रेस पक्ष', 'जनता दाल', 'राष्ट्रीय लोक समता पार्टी', 'झारखंड मुक्ती मोर्चा', 'कम्युनिस्ट', 'मार्क्सवादी', 'हिंदुस्तानी अवाम मोर्चा', 'विकासशील इन्सान पार्टी', 'मुस्लिम लीग', 'जन अधिकारी पार्टी', 'विदुथलाई चिरुथाईगल', 'झारखंड विकास मोर्चा', 'स्वाभिमानी पाक्ष', 'बहुजन विकास आघाडी', 'लेनिनवादी', 'केरळ काँग्रेस', 'समाजवाद', 'समाजवादी पक्ष', 'नेहरू', 'कोंगरेस', 'टीएमसी', 'दीदी', 'भीम', 'जय हिंद', 'हिंद', 'ममता', 'स्टॅलिन', 'कनिमोळी', 'पेरियार', 'पिनराई', 'विजयान', 'मुक्ती', 'मोर्चा', 'विकासशील', 'कोंगुनाडू', 'लालू', 'तेजस्वी', 'उपेंद्र', 'सोरेन', 'येचुरी',
        'کانگریس', 'یو پی اے', 'گاندھی', 'راہول', 'سونیا', 'من موہن', 'پپو', 'دراوڑ منترا کازگم', 'قومی جنتا دال', 'نیشنلسٹ کانگریس پارٹی', 'جنتا دال', 'راشٹریہ لوک سمتا پارٹی', 'جھارکھنڈ مکتی مورچہ', 'کمیونسٹ', 'مارکسی', 'ہندوستانی سول مورچہ', 'وکاسیل انسان پارٹی', 'مسلم لیگ', 'جن ادھیکار پارٹی', 'جھارکھنڈ وکاس مورچہ', 'سوابھیمانی پکشا', 'بہوجن وکاس آغادی', 'لیننسٹ', 'کیرالہ کانگریس', 'سوشلسٹ', 'سوشلسٹ پارٹی', 'نہرو', 'دیدی', 'بھیم', 'جے ہند', 'پچھواڑا', 'مامتا', 'اسٹالن', 'کنیموزی', 'پنارائی', 'وجیان', 'مکتی', 'مورچہ', 'وکاسیل', 'کونگوناڈو', 'لالو', 'تیجسوی', 'اپندر', 'سورن', 'یچوری',
        'કોંગ્રેસ', 'ઉપા', 'નવનીતલાલ', 'રાહુલ', 'સોનિયા', 'મનમોહન', 'પપ્પુ', 'દ્રવિડ મુનેત્ર કઝગમ', 'રાષ્ટ્રીય જનતા દાળ', 'રાષ્ટ્રવાદી કોંગ્રેસ પાર્ટી', 'જનતા દળ', 'રાષ્ટ્રીય લોક સમતા પાર્ટી', 'ઝારખંડ મુક્તિ મોરચા', 'સામ્યવાદી', 'માર્ક્સવાદી', 'હિંદુસ્તાની આદમ મોરચા', 'વિકાસશીલ ઇન્સાન પાર્ટી', 'મુસ્લિમ લીગ', 'જન અધિકાર પાર્ટી', 'વિદુથલાઇ ચિરથૈગલ', 'ઝારખંડ વિકાસ મોરચા', 'સ્વાભિમાની પક્ષ', 'બહુજન વિકાસ આઘાડી', 'લેનિનવાદી', 'કેરળ કોંગ્રેસ', 'સમાજવાદી', 'સમાજવાદી પક્ષ', 'મરુમાલાર્ચી દ્રવિડ મુનેત્ર કઝગમ', 'નેહરુ', 'કોંગ્રેસ', 'દીદી', 'ભીમ,', 'જય હિન્દ', 'મમતા', 'સામ્યવાદી', 'સ્ટાલિન', 'કનિમોઝી', 'પેરિયાર', 'પિનારાઈ', 'વિજયન', 'મુક્તિ', 'મોરચા', 'વિકાસશીલ', 'સ્વાભિમાની પક્ષ', 'કોંગુનાડુ', 'લાલુ', 'તેજસ્વી',
        'ಕಾಂಗ್ರೆಸ್', 'ಉಪ', 'ಗಾಂಧಿ', 'ರಾಹುಲ್', 'ಸೋನಿಯಾ', 'ಮನಮೋಹನ್', 'ಪಪ್ಪು', 'ದ್ರಾವಿಡ ಮುನ್ನೇತ್ರ ಕಳಗಂ', 'ರಾಷ್ಟ್ರೀಯ ಜನತಾ ದಾಲ್', 'ರಾಷ್ಟ್ರೀಯತಾವಾದಿ ಕಾಂಗ್ರೆಸ್ ಪಕ್ಷ', 'ಜನತಾ ದಾಲ್', 'ರಾಷ್ಟ್ರೀಯ ಲೋಕಸಮತಾ ಪಾರ್ಟಿ', 'ಜಾರ್ಖಂಡ್ ಮುಕ್ತಿ ಮೋರ್ಚಾ', 'ಕೋಮುವಾದಿ', 'ಮಾರ್ಕ್ಸ್ವಾದಿ', 'ಹಿಂದೂಸ್ತಾನಿ ಅವಮ್ ಮೋರ್ಚಾ', 'ವಿಕಾಸೀಲ್ ಇನ್ಸಾನ್ ಪಾರ್ಟಿ', 'ಮುಸ್ಲಿಂ ಲೀಗ್', 'ಜನ ಅಧಿಕಾರಿ ಪಾರ್ಟಿ', 'ವಿದುಥಲೈ ಚಿರುಥೈಗಲ್', 'ಜಾರ್ಖಂಡ್ ವಿಕಾಸ್ ಮೋರ್ಚಾ', 'ಸ್ವಾಭಿಮಾನಿ ಪಕ್ಷ', 'ಬಹುಜನ್ ವಿಕಾಸ್ ಆಘಾಡಿ', 'ಲೆನಿನ್ವಾದಿ', 'ಕೇರಳ ಕಾಂಗ್ರೆಸ್', 'ಸಮಾಜವಾದಿ', 'ಸಮಾಜವಾದಿ ಪಕ್ಷ', 'ಮಾರುಮಲಾರ್ಚಿ ದ್ರಾವಿಡ ಮುನ್ನೇತ್ರ ಕಳಗಂ', 'ನೆಹರೂ', 'ಕೊಂಗ್ರೆಸ್', 'ಟಿಎಂಸಿ', 'ದೀ', 'ಭೀಮ್', 'ಜೈ ಹಿಂ', 'ಹಿಂದ್', 'ಮಮತಾ', 'ಸ್ಟಾಲಿನ್', 'ಪೆರಿಯಾರ್', 'ಪಿನಾರೈ', 'ವಿಜಯನ್', 'ಮುಕ್ತಿ', 'ಮೋರ್ಚಾ', 'ವಿಕಾಸ್ ಶೀಲ್', 'ಕೊಂಗುನಾಡು', 'ಲಾಲು', 'ತೇಜಸ್ವಿ', 'ಉಪೇಂದ್ರ', 'ಸೊರೆನ್', 'ಯೆಚೂರಿ',
        'കോൺഗ്രസ്', 'ഉപ', 'ഗാന്ധി', 'രാഹുൽ', 'സോണിയ', 'മൻമോഹൻ', 'പപ്പു', 'ദ്രാവിഡ മുന്നേറ്റ കഴകം', 'രാഷ്ട്രീയ ജനതാ ദാൽ', 'നാഷണലിസ്റ്റ് കോൺഗ്രസ് പാർട്ടി', 'ജനതാ ദാൽ', 'രാഷ്ട്രീയ ലോക് സമത പാർട്ടി', 'ഝാർഖണ്ഡ് മുക്തി മോർച്ച', 'കമ്മ്യൂണിസ്റ്റ്', 'മാർക്സിസ്റ്റ്', 'ഹിന്ദുസ്ഥാനി അവാം മോർച്ച', 'വികാശീൽ ഇൻസാൻ പാർട്ടി', 'മുസ്ലീം ലീഗ്', 'ജൻ അധികാർ പാർട്ടി', 'വിദുതലൈ ചിരുതൈഗൽ', 'ജാർഖണ്ഡ് വികാസ് മോർച്ച', 'സ്വാഭിമാനി പക്ഷ', 'ബാഹുജൻ വികാസ് ആഘാദി', 'ലെനിനിസ്റ്റ്', 'കേരള കോൺഗ്രസ്', 'സോഷ്യലിസ്റ്റ്', 'സോഷ്യലിസ്റ്റ് പാർട്ടി', 'മരുമലാർച്ചി ദ്രാവിഡ മുന്നേത്ര കഴകം', 'നെഹ്രു', 'കോംഗ്രെസ്', 'ദീദി', 'ഭീം', 'ജയ് ഹിന്ദ്', 'പുറകോട്ട്', 'മംത', 'സ്റ്റാലിൻ', 'കനിമൊഴി', 'പെരിയാർ', 'പിനരൈ', 'വിജയൻ', 'മുക്തി', 'മോർച്ച', 'വികാസ്ഷീൽ', 'കൊങ്കുനാട്', 'ലാലു', 'തേജസ്വി', 'ഉപേന്ദ്ര', 'സോറൻ', 'യെച്ചൂരി',
        'ਕਾਂਗਰਸ', 'ਉਪਾ', 'ਗਾਂਧੀ', 'ਰਾਹੁਲ', 'ਸੋਨੀਆ', 'ਮਨਮੋਹਨ', 'ਪੱਪੂ', 'ਦ੍ਰਾਵਿਦਾ ਮੁਨੇਟਰਾ ਕਾਜਗਮ', 'ਰਾਸ਼ਟਰੀ ਜਨਤਾ ਦਾਲ', 'ਰਾਸ਼ਟਰਵਾਦੀ ਕਾਂਗਰਸ ਪਾਰਟੀ', 'ਜਨਤਾ ਦਾਲ', 'ਰਾਸ਼ਟਰੀ ਲੋਕ ਸਮਤਾ ਪਾਰਟੀ', 'ਝਾਰਖੰਡ ਮੁਕਤੀ ਮੋਰਚਾ', 'ਕਮਿਊਨਿਸਟ', 'ਮਾਰਕਸਵਾਦੀ', 'ਹਿੰਦੁਸਤਾਨੀ ਆਵਾਮ ਮੋਰਚਾ', 'ਵਿਕਾਸ਼ੀਲ ਇਨਸਾਨ ਪਾਰਟੀ', 'ਮੁਸਲਿਮ ਲੀਗ', 'ਜਨ ਅਧਿਕਾਰੀ ਪਾਰਟੀ', 'ਵਿਡੁਥਲਾਈ ਚਿਰੂਥਾਈਗਲ', 'ਝਾਰਖੰਡ ਵਿਕਾਸ ਮੋਰਚਾ', 'ਸਵਾਭਿਮਾਨੀ ਪਕਸ਼ਾ', 'ਬਹੁਜਨ ਵਿਕਾਸ ਅਗਾੜੀ', 'ਲੈਨਿਨਿਸਟ', 'ਕੇਰਲਾ ਕਾਂਗਰਸ', 'ਸਮਾਜਵਾਦੀ', 'ਸਮਾਜਵਾਦੀ ਪਾਰਟੀ', 'ਮਾਰੂਮਲਾਰਚੀ ਦ੍ਰਾਵਿਦਾ ਮੁਨੇਤਰਾ ਕਾਜਗਮ', 'ਨਹਿਰੂ', 'ਕੋਂਗਰੇਸ', 'ਦੀਦੀ', 'ਭੀਮ', 'ਜੈ ਹਿੰਦ', 'ਹਿੰਦ', 'ਮਮਤਾ', 'ਸਟਾਲਿਨ', 'ਕਨੀਮੋਜ਼ੀ', 'ਪੈਰੀਅਰ', 'ਪਿਨਾਰਾਈ', 'ਵਿਜਯਾਨ', 'ਮੁਕਤਿ', 'ਮੋਰਚਾ', 'ਵਿਕਾਸਸ਼ੀਲ', 'ਕੋਂਗਨਾਡੂ', 'ਲਾਲੂ', 'ਤੇਜਸਵੀ', 'ਉਪੇਂਦਰ', 'ਯੇਚੁਰੀ',
        'କଂଗ୍ରେସ', 'ଗାନ୍ଧୀ', 'ରାହୁଲ', 'ସୋନିଆ', 'ମନମୋହନ', 'ପପୁ', 'ଦ୍ରାବିଡା ମୁନ୍ନେଟ୍ରା କାଜାଗମ୍ |', 'ରାଷ୍ଟ୍ରୀୟ ଜନତା ଡାଲ', 'ଜାତୀୟତାବାଦୀ କଂଗ୍ରେସ ପାର୍ଟି', 'ଜନତା ଡାଲ', 'ରାଷ୍ଟ୍ରୀୟ ଲୋକ ସମତା ପାର୍ଟି', 'ଝାଡ଼ଖଣ୍ଡ ମୁକ୍ତି ମୋର୍ଚ୍ଚା', 'କମ୍ୟୁନିଷ୍ଟ', 'ମାର୍କ୍ସବାଦୀ', 'ହିନ୍ଦୁସ୍ତାନୀ ଆୱାମ ମୋର୍ଚ୍ଚା', 'ବିକଶିତ ଇନ୍ ସାନ୍ ପାର୍ଟି', 'ମୁସଲିମ ଲିଗ', 'ଜନ ଆଧିକର ପାର୍ଟି', 'ଝାଡ଼ଖଣ୍ଡ ବିକାଶ ମୋର୍ଚ୍ଚା', 'ଏଲିନା କହିଲ', 'କେରଳ କଂଗ୍ରେସ'
        ]

    bjp_flags = []
    ing_flags = []

    bjp_keywords = [keyword for keyword in bjp_keywords]
    ing_keywords = [keyword for keyword in ing_keywords]

    for index, row in sourcedata.iterrows():
        text = row['comment_textDisplay']

        bjp_flag = 1 if any(keyword in text for keyword in bjp_keywords) else 0
        bjp_flags.append(bjp_flag)

        ing_flag = 1 if any(keyword in text for keyword in ing_keywords) else 0
        ing_flags.append(ing_flag)

    sourcedata['bjp'] = bjp_flags
    sourcedata['ing'] = ing_flags
    return sourcedata
################################################################################################
#Step18: Remove comments which doesnt attribute to either BJP or Congress
def RemoveCommentswithallFlags0(sourcedata):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes the comments which doesn't attribute to both parties
    :param sourcedata: data frame
    :type sourcedata: data frame
    :return: data frame
    """
    # Drop rows where both "bjp" and "ing" columns have 0 values
    validpartiesdf = sourcedata[(sourcedata['bjp'] != 0) | (sourcedata['ing'] != 0)]
    return validpartiesdf
################################################################################################
#Step19: Remove comments which doesnt attribute to either BJP or Congress
def BlankCommentsRemoval(sourcedata, columnname):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function removes the blank comments
    :param sourcedata: data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :return: data frame
    """
    sourcedata = sourcedata[sourcedata[columnname].str.strip() != '']  # Filter out rows where the column is not blank
    return sourcedata
################################################################################################
#Step20: Yoytube comments data collected is doesnt have sentiment label or score in it.
#To find the sentiment of the comments, we are using unsupervised approach using mBERT multilingual pretrained base model which consider uncased/not case sensitive
#If the source data already had these labels, we could move on to model build and prediction directly
def Compute_polarity_score_mBERT(sourcedata, columnname, langColumn):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function use unsupervised approach using mBERT multilingual pretrained base model to get sentiment
    :param sourcedata: data frame
    :type sourcedata: data frame
    :param columnname: The comments text column
    :type columnname: str
    :param langColumn: The column with language code in it
    :type langColumn: str
    :return: data frame
    """
    distinct_langcodes = sourcedata[langColumn].unique()

    model_lang_tokenizer_map = {
        "en": "bert-base-multilingual-uncased",
        "hi": "bert-base-multilingual-uncased",
        "bn": "bert-base-multilingual-uncased",
        "te": "bert-base-multilingual-uncased",
        "ta": "bert-base-multilingual-uncased",
        "mr": "bert-base-multilingual-uncased",
        "ur": "bert-base-multilingual-uncased",
        "gu": "bert-base-multilingual-uncased",
        "kn": "bert-base-multilingual-uncased",
        "ml": "bert-base-multilingual-uncased",
        "pa": "bert-base-multilingual-uncased",
        "or": "bert-base-multilingual-uncased"
    }

    def compute_polarity(text, tokenizer, model):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        scores = torch.softmax(logits, dim=1)
        return scores[0].tolist()

    polarity_scored_df = pd.DataFrame(columns=sourcedata.columns)

    for language_code in distinct_langcodes:
        if language_code in model_lang_tokenizer_map:
            model_name = model_lang_tokenizer_map[language_code]
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
            language_df = sourcedata[sourcedata[langColumn] == language_code].copy()
            language_df[["positive_score", "negative_score", "neutral_score"]] = language_df[columnname].apply(lambda x: pd.Series(compute_polarity(x, tokenizer, model)))
            polarity_scored_df = pd.concat([polarity_scored_df, language_df], ignore_index=True)
        else:
            print(f"NLP mBERT model not found for language: {language_code}")
    return polarity_scored_df
################################################################################################
#Step21: Based on polarity score compute the sentiment by finding max of all 3 classes
def compute_sentiments(scorerecord):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function will classify the sentiment based on 3 class polarity scores
    :param scorerecord: The row record with polarity scores
    :type scorerecord: str
    :return: str
    """
    if scorerecord["positive_score"] > scorerecord["negative_score"] and scorerecord["positive_score"] > scorerecord["neutral_score"]:
        return "Positive"
    elif scorerecord["negative_score"] > scorerecord["positive_score"] and scorerecord["negative_score"] > scorerecord["neutral_score"]:
        return "Negative"
    else:
        return "Neutral"
################################################################################################
#Step22: Train NLP Multilingual mBERT by looping languages for each iteration
#Note: 11 Languages are considered for this model development
#BASE MODEL - WITH ONLY DEFAULTS AND NO OPTIMIZER OR FINETUNING PARAMETERS USED. IT IS PURELY FOR BENCHMARKING
def NLP_BASEMODEL_LANGUAGES_mBERT(sourcedata, batch_size, num_epochs, num_classes):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function will classify the sentiment based on 3 class polarity scores
    :param sourcedata: data frame
    :type sourcedata: data frame
    :param batch_size: The batch size to process
    :type batch_size: int
    :param num_epochs: The iterations to run
    :type num_epochs: int
    :param num_classes: The number of sentiment classes
    :type num_classes: int
    :return: data frame
    """
    Distinct_Languages = sourcedata['language_code'].unique()

    model_tokenizer_mapping = {
        "en": "bert-base-multilingual-uncased",
        "hi": "bert-base-multilingual-uncased",
        "bn": "bert-base-multilingual-uncased",
        "te": "bert-base-multilingual-uncased",
        "ta": "bert-base-multilingual-uncased",
        "mr": "bert-base-multilingual-uncased",
        "ur": "bert-base-multilingual-uncased",
        "gu": "bert-base-multilingual-uncased",
        "kn": "bert-base-multilingual-uncased",
        "ml": "bert-base-multilingual-uncased",
        "pa": "bert-base-multilingual-uncased",
        "or": "bert-base-multilingual-uncased"
    }

    metrics_dict = {
        'ModelName': [],  # New column for model name
        'LanguageCode': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1Score': []
    }

    for language_code in Distinct_Languages:
        #model and tokenizer name for language code
        model_name = model_tokenizer_mapping.get(language_code, 'bert-base-multilingual-cased')
        language_df = sourcedata[sourcedata['language_code'] == language_code]
        train_df, test_df = train_test_split(language_df, test_size=0.3, random_state=42)

        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

        label_mapping = {"Positive": 0, "Negative": 1, "Neutral": 2}

        train_labels_numeric = [label_mapping[label] for label in train_df['mBert_sentiment']]
        test_labels_numeric = [label_mapping[label] for label in test_df['mBert_sentiment']]

        #Convert labels to one-hot encoding to run mBERT
        def one_hot_encode_labels(labels, num_classes):
            one_hot_labels = []
            for label in labels:
                one_hot = [0] * num_classes
                one_hot[label] = 1
                one_hot_labels.append(one_hot)
            return torch.tensor(one_hot_labels, dtype=torch.float32)

        train_labels = one_hot_encode_labels(train_labels_numeric, num_classes)
        test_labels = one_hot_encode_labels(test_labels_numeric, num_classes)

        #Preparing input data for BERT inputs
        train_encodings = tokenizer(list(train_df['comment_textDisplay']), truncation=True, padding=True, max_length=128,
                                    return_tensors='pt')
        test_encodings = tokenizer(list(test_df['comment_textDisplay']), truncation=True, padding=True, max_length=128, return_tensors='pt')

        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        #Training for languages loop
        for epoch in range(num_epochs):
            print("#")
            print(f"Language code {language_code}: Epoch {epoch + 1}/{num_epochs} is running...")
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                train_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {train_loss / len(train_dataloader)}")

        #Evaluation on test dataset
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        test_labels_decoded = [np.argmax(label) for label in test_labels.cpu().numpy()]

        accuracy = accuracy_score(test_labels_decoded, predictions)
        precision = precision_score(test_labels_decoded, predictions, average='weighted', zero_division=1)
        recall = recall_score(test_labels_decoded, predictions, average='weighted', zero_division=1)
        f1 = f1_score(test_labels_decoded, predictions, average='weighted')

        metrics_dict['ModelName'].append('mBERT Base Model')
        metrics_dict['LanguageCode'].append(language_code)
        metrics_dict['Accuracy'].append(accuracy)
        metrics_dict['Precision'].append(precision)
        metrics_dict['Recall'].append(recall)
        metrics_dict['F1Score'].append(f1)
    baseModel_Eval_metrics = pd.DataFrame(metrics_dict)
    return baseModel_Eval_metrics

################################################################################################
#Step23: Train NLP Multilingual mBERT by looping languages for each iteration
#Note: 11 Languages are considered for this model development and validation
#FINETUNED MODEL - WITH OPTIMIZER AND FINETUNING PARAMETERS USED.
def NLP_FINETUNEDMODEL_LANGUAGES_mBERT(sourcedata, batch_size, num_epochs, num_classes, learning_rate):
    """
    Author: Jagadeesan Rajalakshmi Vellaichamy
    Reviewer: Dani Papamaximou
    Created At: 20/08/2023
    Description: This function will classify the sentiment based on 3 class polarity scores
    :param sourcedata: data frame
    :type sourcedata: data frame
    :param batch_size: The batch size to process
    :type batch_size: int
    :param num_epochs: The iterations to run
    :type num_epochs: int
    :param num_classes: The number of sentiment classes
    :type num_classes: int
    :return: data frame
    """
    Distinct_Languages = sourcedata['language_code'].unique()

    # Define a mapping of language codes to model names
    model_tokenizer_mapping = {
        "en": "bert-base-multilingual-uncased",
        "hi": "bert-base-multilingual-uncased",
        "bn": "bert-base-multilingual-uncased",
        "te": "bert-base-multilingual-uncased",
        "ta": "bert-base-multilingual-uncased",
        "mr": "bert-base-multilingual-uncased",
        "ur": "bert-base-multilingual-uncased",
        "gu": "bert-base-multilingual-uncased",
        "kn": "bert-base-multilingual-uncased",
        "ml": "bert-base-multilingual-uncased",
        "pa": "bert-base-multilingual-uncased",
        "or": "bert-base-multilingual-uncased"
    }

    metrics_dict = {
        'ModelName': [],
        'LanguageCode': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1Score': []
    }

    for language_code in Distinct_Languages:
        model_name = model_tokenizer_mapping.get(language_code, 'bert-base-multilingual-cased')
        language_df = sourcedata[sourcedata['language_code'] == language_code]
        train_df, test_df = train_test_split(language_df, test_size=0.3, random_state=42)

        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

        label_mapping = {"Positive": 0, "Negative": 1, "Neutral": 2}

        train_labels_numeric = [label_mapping[label] for label in train_df['mBert_sentiment']]
        test_labels_numeric = [label_mapping[label] for label in test_df['mBert_sentiment']]

        #Convert labels to one-hot encoding for mBERT
        def one_hot_encode_labels(labels, num_classes):
            one_hot_labels = []
            for label in labels:
                one_hot = [0] * num_classes
                one_hot[label] = 1
                one_hot_labels.append(one_hot)
            return torch.tensor(one_hot_labels, dtype=torch.float32)

        train_labels = one_hot_encode_labels(train_labels_numeric, num_classes)
        test_labels = one_hot_encode_labels(test_labels_numeric, num_classes)

        #Data preparation for BERT inputs
        train_encodings = tokenizer(list(train_df['comment_textDisplay']), truncation=True, padding=True, max_length=128,
                                    return_tensors='pt')
        test_encodings = tokenizer(list(test_df['comment_textDisplay']), truncation=True, padding=True, max_length=128, return_tensors='pt')

        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        #optimizer and loss function - fine tuning
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        #Training for loop
        for epoch in range(num_epochs):
            print("#")
            print(f"Language code {language_code}: Epoch {epoch + 1}/{num_epochs} is running...")
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                labels = batch[2].argmax(dim=1).to(device)  #Convert one-hot labels to class indices
                optimizer.zero_grad()
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {train_loss / len(train_dataloader)}")

        #scoring test dataset - Evaluation
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        test_labels_decoded = [np.argmax(label) for label in test_labels.cpu().numpy()]

        accuracy = accuracy_score(test_labels_decoded, predictions)
        precision = precision_score(test_labels_decoded, predictions, average='weighted', zero_division=1)
        recall = recall_score(test_labels_decoded, predictions, average='weighted', zero_division=1)
        f1 = f1_score(test_labels_decoded, predictions, average='weighted')

        metrics_dict['ModelName'].append('mBERT Finetuned Model')
        metrics_dict['LanguageCode'].append(language_code)
        metrics_dict['Accuracy'].append(accuracy)
        metrics_dict['Precision'].append(precision)
        metrics_dict['Recall'].append(recall)
        metrics_dict['F1Score'].append(f1)

    FinetunedModel_Eval_metrics = pd.DataFrame(metrics_dict)
    return FinetunedModel_Eval_metrics

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_date = '2019-01-01'
    end_date = '2019-04-10'
    # data = pd.read_csv("C:\\Dissertation_2023\\youtube_comments\\youtube_apidata_47.csv", sep=',')
    data = FileReadFromDirectory("C:\\Dissertation_2023\\youtube_comments\\", "youtube_apidata_*.csv")
    data = AnalysisWindowTimePeriodFilter(data, start_date, end_date, "ytvideo_publishedAt")
    data = SmileyConversiontoTexts(data, "comment_textDisplay")
    data['comment_textDisplay'] = data['comment_textDisplay'].apply(EmojiRemovalfromComments)
    data = Remove_NAs_Blanks(data, "comment_textDisplay")
    data = Punctuations_Removal(data, "comment_textDisplay")
    data = DuplicateCommentsRemoval(data, "comment_textDisplay")
    data = Language_Identification(data, 'comment_textDisplay')
    data = Unidentified_language_removal(data)
    data = SinglegramComments_Removal(data, 'comment_textDisplay')
    data = NumbersinComments_Removal(data, 'comment_textDisplay')
    data = RepeatwordsInCommentsRemoval(data, 'comment_textDisplay')
    data_eng = data[data['language_code'] == 'en']
    data_eng = data_eng.apply(Custom_language_detection, axis=1)
    data_eng = data_eng.apply(Custom_language_code_mapping, axis=1)
    data_eng = English_comments_to_lower(data_eng, 'comment_textDisplay')
    data_eng['comment_textDisplay'] = data_eng['comment_textDisplay'].apply(lambda x: Stopwords_detection_removal(x))
    data_noneng = data[data['language_code'] != 'en']
    final = pd.concat([data_eng, data_noneng], ignore_index=True)
    final = CreateFlagsbyLabelingParty(final)
    final = RemoveCommentswithallFlags0(final)  # Removing comments which has flag values bjp=0 and ing=0
    final = BlankCommentsRemoval(final, 'comment_textDisplay')
    final = Compute_polarity_score_mBERT(final, "comment_textDisplay", "language_code")
    final["mBert_sentiment"] = final.apply(compute_sentiments, axis=1)
    final.to_csv("C:\\Dissertation_2023\\Youtube_Clean_dataframe.csv", index=False)
    mBERTbaseModel_metrics = NLP_BASEMODEL_LANGUAGES_mBERT(final, 2, 1, 3)
    mBERTFitModel_metrics = NLP_FINETUNEDMODEL_LANGUAGES_mBERT(final, 2, 1, 3, 2e-5)
    mbert_lang_eva_metrics = pd.concat([mBERTbaseModel_metrics, mBERTFitModel_metrics], ignore_index=True)
    mbert_lang_eva_metrics.to_csv("C:\\Dissertation_2023\\NLP_mBERT_Metrics.csv", index=False)
