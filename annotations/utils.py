import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import emoji
from laserembeddings import Laser
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


laser = Laser()
indian_sbert = SentenceTransformer('../multilingual-sbert/models/se-asian-sbert')
portuguese_sbert = SentenceTransformer('distiluse-base-multilingual-cased')
english_sbert = SentenceTransformer('bert-base-nli-mean-tokens')

uchr = chr  # Python 3

# Unicode 11.0 Emoji Component map (deemed safe to remove)
_removable_emoji_components = (
    (0x20E3, 0xFE0F),  # combining enclosing keycap, VARIATION SELECTOR-16
    range(0x1F1E6, 0x1F1FF + 1),  # regional indicator symbol letter a..regional indicator symbol letter z
    range(0x1F3FB, 0x1F3FF + 1),  # light skin tone..dark skin tone
    range(0x1F9B0, 0x1F9B3 + 1),  # red-haired..white-haired
    range(0xE0020, 0xE007F + 1),  # tag space..cancel tag
)
emoji_components = re.compile(u'({})'.format(u'|'.join([
    re.escape(uchr(c)) for r in _removable_emoji_components for c in r])),
    flags=re.UNICODE)


def remove_emoji(text, remove_components=True):
    cleaned = emoji.get_emoji_regexp().sub(u'', text)
    if remove_components:
        cleaned = emoji_components.sub(u'', cleaned)
    return cleaned


spam_list = {
    'en': ['free', 'salary', 'job', 'iphone', 'unlocked', 'samsung', 'qualification', 'ielts', 'sex', 'click', 'galaxy',
           'sales', 'vacancy', 'call', 'hot', 'app', 'buy', 'buyers', 'guaranteed', 'subscribe', 'betting', 'cash',
           'price', 'invest', 'mobile', 'offer', 'sale', 'warranty', 'brand', 'girls', 'guarantee', 'qualificayions',
           'bestjob', 'consultant', 'horny', 'toefl', 'visa', 'workcall', 'certificate', 'donate', 'earn', 'money',
           'passport', 'porno', 'discount', 'officework', 'sexy', 'exam', 'futureopportunity', 'games', 'paytm',
           'products', 'redmi', 'referral', 'registration', 'unlimited', 'win', 'won', 'amzn', 'bank', 'females',
           'hurry', 'income', 'investment', 'mba', 'opportunities', 'opportunity', 'paying', 'wanted', 'accessories',
           'adults', 'adultsgroup', 'appbrowzer', 'gadgets', 'infobrand', 'mobiles', 'offers', 'onlybrand', 'packages',
           'paid', 'payment', 'porn', 'promo', 'referrals', 'referrer', 'register'],
    'hi': ['नि: शुल्क', 'वेतन', 'काम', 'आई - फ़ोन', 'अनलॉक हो गया है', 'सैमसंग', 'योग्यता', 'आईईएलटीएस', 'लिंग',
           'क्लिक', 'आकाशगंगा', 'बिक्री', 'रिक्ति', 'कॉल', 'गरम', 'एप्लिकेशन', 'खरीद', 'खरीददारों', 'गारंटी',
           'सदस्यता लेने के', 'शर्त', 'नकद', 'कीमत', 'निवेश', 'मोबाइल', 'प्रस्ताव', 'बिक्री', 'गारंटी', 'ब्रांड',
           'लड़कियाँ', 'गारंटी', 'qualificayions', 'bestjob', 'सलाहकार', 'सींग का बना हुआ', 'टॉफेल', 'वीसा', 'workcall',
           'प्रमाणपत्र', 'दान करना', 'कमाना', 'पैसे', 'पासपोर्ट', 'अश्लील', 'छूट', 'कार्यालय का काम', 'कामुक',
           'परीक्षा', 'futureopportunity', 'खेल', 'पेटीएम', 'उत्पादों', 'redmi', 'रेफरल', 'पंजीकरण', 'असीमित', 'जीत',
           'जीत लिया', 'AMZN', 'बैंक', 'महिलाओं', 'जल्दी कीजिये', 'आय', 'निवेश', 'एमबीए', 'अवसरों', 'अवसर', 'का भुगतान',
           'चाहता था', 'सामान', 'वयस्कों', 'adultsgroup', 'appbrowzer', 'गैजेट', 'infobrand', 'मोबाइल्स', 'प्रस्तावों',
           'onlybrand', 'संकुल', 'भुगतान किया है', 'भुगतान', 'पॉर्न', 'प्रोमो', 'रेफरल', 'रेफरर', 'रजिस्टर करें'],
    'hi-Latn': ['free', 'salary', 'job', 'iphone', 'unlocked', 'samsung', 'qualification', 'ielts', 'sex', 'click', 'galaxy',
           'sales', 'vacancy', 'call', 'hot', 'app', 'buy', 'buyers', 'guaranteed', 'subscribe', 'betting', 'cash',
           'price', 'invest', 'mobile', 'offer', 'sale', 'warranty', 'brand', 'girls', 'guarantee', 'qualificayions',
           'bestjob', 'consultant', 'horny', 'toefl', 'visa', 'workcall', 'certificate', 'donate', 'earn', 'money',
           'passport', 'porno', 'discount', 'officework', 'sexy', 'exam', 'futureopportunity', 'games', 'paytm',
           'products', 'redmi', 'referral', 'registration', 'unlimited', 'win', 'won', 'amzn', 'bank', 'females',
           'hurry', 'income', 'investment', 'mba', 'opportunities', 'opportunity', 'paying', 'wanted', 'accessories',
           'adults', 'adultsgroup', 'appbrowzer', 'gadgets', 'infobrand', 'mobiles', 'offers', 'onlybrand', 'packages',
           'paid', 'payment', 'porn', 'promo', 'referrals', 'referrer', 'register'],
    'ta': ['இலவசம்', 'சம்பளம்', 'வேலை', 'ஐபோன்', 'திறக்கப்பட்டது', 'சாம்சங்', 'தகுதி', 'ielts', 'செக்ஸ்',
           'கிளிக் செய்க', 'விண்மீன்', 'விற்பனை', 'காலியிடம்', 'அழைப்பு', 'சூடான', 'செயலி', 'வாங்க', 'வாங்குபவர்கள்',
           'உத்தரவாதம்', 'பதிவு', 'பந்தயம்', 'பணம்', 'விலை', 'முதலீடு', 'கைபேசி', 'சலுகை', 'விற்பனை', 'உத்தரவாதத்தை',
           'பிராண்ட்', 'பெண்கள்', 'உத்தரவாதம்', 'தகுதிகள்', 'bestjob', 'ஆலோசகர்', 'கொம்பு', 'toefl', 'விசா',
           'பணிக்குழு', 'சான்றிதழ்', 'நன்கொடை', 'சம்பாதி', 'பணம்', 'கடவுச்சீட்டு', 'porno', 'தள்ளுபடி', 'அலுவலக வேலை',
           'கவர்ச்சியாக', 'தேர்வு', 'எதிர்கால வாய்ப்பு', 'விளையாட்டுகள்', 'paytm', 'தயாரிப்புகள்', 'ரெட்மி',
           'பரிந்துரை', 'பதிவு', 'வரம்பற்ற', 'வெற்றி', 'வென்றது', 'amzn', 'வங்கி', 'பெண்கள்', 'அவசரம்', 'வருமானம்',
           'முதலீடு', 'mba', 'வாய்ப்புகள்', 'வாய்ப்பு', 'செலுத்துதல்', 'விரும்பினார்', 'பாகங்கள்', 'பெரியவர்கள்',
           'பெரியவர்கள் குழு', 'appbrowzer', 'கேஜெட்டுகள்', 'infobrand', 'மொபைல்கள்', 'சலுகைகள்', 'ஒரே பிராண்ட்',
           'தொகுப்புகள்', 'செலுத்தப்பட்டது', 'கட்டணம்', 'ஆபாச', 'விளம்பர', 'பரிந்துரைகள்', 'பரிந்துரைப்பவர்', 'பதிவு'],
    'te': ['ఉచిత', 'జీతం', 'ఉద్యోగం', 'ఐఫోన్', 'అన్లాక్', 'శామ్సంగ్', 'క్వాలిఫికేషన్', 'ఐఇఎల్టిఎస్', 'సెక్స్', 'క్లిక్',
           'గెలాక్సీ', 'అమ్మకాలు', 'ఖాళీ', 'కాల్', 'వేడి', 'అనువర్తనం', 'కొనుగోలు', 'కొనుగోలుదారులు', 'హామీ', 'చందా',
           'బెట్టింగ్', 'నగదు', 'ధర', 'పెట్టుబడి', 'మొబైల్', 'ఆఫర్', 'అమ్మకానికి', 'వారంటీ', 'బ్రాండ్', 'అమ్మాయిలు',
           'హామీ', 'qualificayions', 'bestjob', 'కన్సల్టెంట్', 'horny', 'TOEFL', 'వీసా', 'workcall', 'సర్టిఫికేట్',
           'దానం', 'సంపాదించడానికి', 'డబ్బు', 'పాస్పోర్ట్', 'పోర్నో', 'డిస్కౌంట్', 'కార్యాలయ పని', 'సెక్సీ', 'పరీక్షలో',
           'futureopportunity', 'ఆటలు', 'Paytm', 'ఉత్పత్తులు', 'redmi', 'రెఫరల్', 'నమోదు', 'అపరిమిత', 'గెలుపు',
           'గెలిచింది', 'amzn', 'బ్యాంకు', 'ఆడ', 'అత్యవసరము', 'ఆదాయం', 'పెట్టుబడి', 'MBA', 'అవకాశాలు', 'అవకాశం',
           'చెల్లించి', 'కావలెను', 'ఉపకరణాలు', 'పెద్దలు', 'adultsgroup', 'appbrowzer', 'గాడ్జెట్లు', 'infobrand',
           'మొబైల్', 'ఆఫర్లు', 'onlybrand', 'ప్యాకేజీలు', 'చెల్లించిన', 'చెల్లింపు', 'శృంగార', 'ప్రోమో', 'పంపండి',
           'నివేదనకు', 'నమోదు'],
    'ml': ['സൗ ജന്യം', 'ശമ്പളം', 'ജോലി', 'ഐഫോൺ', 'അൺലോക്ക്', 'സാംസങ്', 'യോഗത', 'ഇഎല്ത്സ്', 'ലിംഗം', 'ക്ലിക്കിൽ',
           'ഗാലക്സി', 'വിൽപ്പന', 'ഒഴിവ്', 'വിളി', 'ചൂടുള്ള', 'അപ്ലിക്കേഷൻ', 'വാങ്ങാൻ', 'വാങ്ങലുകാരെ', 'ഗ്യാരണ്ടി',
           'സബ്സ്ക്രൈബ്', 'വാതുവയ്പ്പ്', 'പണം', 'വില', 'നിക്ഷേപിക്കുക', 'മൊബൈൽ', 'വാഗ്ദാനം', 'വില്പനയ്ക്ക്', 'ഉറപ്പ്',
           'ബ്രാൻഡ്', 'പെൺകുട്ടികൾ', 'ഗ്യാരണ്ടി', 'കുഅലിഫിചയിഒംസ്', 'ബെസ്ത്ജൊബ്', 'കൂടിയാലോചിക്കുന്നവള്',
           'അച്യുതാനന്ദന്', 'ഉപ്പാപ്പക്ക്', 'വിസ', 'വൊര്ക്ചല്ല്', 'സർട്ടിഫിക്കറ്റ്', 'സംഭാവനചെയ്യുക', 'നേടാൻ', 'പണം',
           'പാസ്പോര്ട്ട്', 'അശ്ലീലമായ', 'കുറഞ്ഞ', 'ഓഫീസ് ജോലി', 'സെക്സി', 'പരീക്ഷ', 'ഫുതുരെഒപ്പൊര്തുനിത്യ്',
           'ഗെയിമുകൾ', 'Paytm', 'ഉൽപ്പന്നങ്ങൾ', 'രെദ്മി', 'റഫറൽ', 'രജിസ്ട്രേഷൻ', 'പരിമിതികളില്ലാത്ത', 'വിജയം',
           'ജയിച്ചു', 'AMZN', 'ബാങ്ക്', 'പെൺ', 'ധൃതികൂട്ടുക', 'വരുമാനം', 'നിക്ഷേപം', 'എംബിഎ', 'അവസരങ്ങൾ', 'അവസരം',
           'അടയ്ക്കേണ്ട', 'ആഗ്രഹിച്ചു', 'സാധനങ്ങൾ', 'മുതിർന്നവർ', 'അദുല്ത്സ്ഗ്രൊഉപ്', 'അപ്പ്ബ്രൊവ്ജെര്', 'ഗാഡ്ജറ്റുകൾ',
           'ഇന്ഫൊബ്രംദ്', 'മൊബൈൽ', 'ഓഫറുകൾ', 'ഒംല്യ്ബ്രംദ്', 'പാക്കേജുകൾ', 'പണം', 'പേയ്മെന്റ്', 'അശ്ലീല', 'പ്രൊമോ',
           'ഇതിന്റെ', 'റഫറർ', 'പട്ടിക'],
    'mr': ['नि: शुल्क', 'वेतन', 'काम', 'आई - फ़ोन', 'अनलॉक हो गया है', 'सैमसंग', 'योग्यता', 'आईईएलटीएस', 'लिंग',
           'क्लिक', 'आकाशगंगा', 'बिक्री', 'रिक्ति', 'कॉल', 'गरम', 'एप्लिकेशन', 'खरीद', 'खरीददारों', 'गारंटी',
           'सदस्यता लेने के', 'शर्त', 'नकद', 'कीमत', 'निवेश', 'मोबाइल', 'प्रस्ताव', 'बिक्री', 'गारंटी', 'ब्रांड',
           'लड़कियाँ', 'गारंटी', 'qualificayions', 'bestjob', 'सलाहकार', 'सींग का बना हुआ', 'टॉफेल', 'वीसा', 'workcall',
           'प्रमाणपत्र', 'दान करना', 'कमाना', 'पैसे', 'पासपोर्ट', 'अश्लील', 'छूट', 'कार्यालय का काम', 'कामुक',
           'परीक्षा', 'futureopportunity', 'खेल', 'पेटीएम', 'उत्पादों', 'redmi', 'रेफरल', 'पंजीकरण', 'असीमित', 'जीत',
           'जीत लिया', 'AMZN', 'बैंक', 'महिलाओं', 'जल्दी कीजिये', 'आय', 'निवेश', 'एमबीए', 'अवसरों', 'अवसर', 'का भुगतान',
           'चाहता था', 'सामान', 'वयस्कों', 'adultsgroup', 'appbrowzer', 'गैजेट', 'infobrand', 'मोबाइल्स', 'प्रस्तावों',
           'onlybrand', 'संकुल', 'भुगतान किया है', 'भुगतान', 'पॉर्न', 'प्रोमो', 'रेफरल', 'रेफरर', 'रजिस्टर करें'],
    'bn': ['இலவசம்', 'சம்பளம்', 'வேலை', 'ஐபோன்', 'திறக்கப்பட்டது', 'சாம்சங்', 'தகுதி', 'IELTS', 'செக்ஸ்',
           'கிளிக் செய்க', 'விண்மீன்', 'விற்பனை', 'காலியிடம்', 'அழைப்பு', 'சூடான', 'செயலி', 'வாங்க', 'வாங்குபவர்கள்',
           'உத்தரவாதம்', 'பதிவு', 'பந்தயம்', 'பணம்', 'விலை', 'முதலீடு', 'கைபேசி', 'சலுகை', 'விற்பனை', 'உத்தரவாதத்தை',
           'பிராண்ட்', 'பெண்கள்', 'உத்தரவாதம்', 'தகுதிகள்', 'bestjob', 'ஆலோசகர்', 'கொம்பு', 'TOEFL', 'விசா',
           'பணிக்குழு', 'சான்றிதழ்', 'நன்கொடை', 'சம்பாதி', 'பணம்', 'கடவுச்சீட்டு', 'পর্ণ', 'தள்ளுபடி', 'அலுவலக வேலை',
           'கவர்ச்சியாக', 'தேர்வு', 'எதிர்கால வாய்ப்பு', 'விளையாட்டுகள்', 'Paytm', 'தயாரிப்புகள்', 'ரெட்மி',
           'பரிந்துரை', 'பதிவு', 'வரம்பற்ற', 'வெற்றி', 'வென்றது', 'amzn', 'வங்கி', 'பெண்கள்', 'அவசரம்', 'வருமானம்',
           'முதலீடு', 'এমবিএ', 'வாய்ப்புகள்', 'வாய்ப்பு', 'செலுத்துதல்', 'விரும்பினார்', 'பாகங்கள்', 'பெரியவர்கள்',
           'பெரியவர்கள் குழு', 'appbrowzer', 'கேஜெட்டுகள்', 'infobrand', 'மொபைல்கள்', 'சலுகைகள்', 'ஒரே பிராண்ட்',
           'தொகுப்புகள்', 'செலுத்தப்பட்டது', 'கட்டணம்', 'ஆபாச', 'விளம்பர', 'பரிந்துரைகள்', 'பரிந்துரைப்பவர்', 'பதிவு']}


def convert_from_hindi_latin(text):
    return transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)


url_regex = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
# this regex should only be used to identify whether a phone number exists, it doesn't work well for extracting the number
phone_no_regex = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')


def remove_urls(text):
    while True:
        result = url_regex.search(text)
        if result == None:
            break
        text = text.replace(result.group(), '<URL>')
    return text


def contains_url(text):
    return url_regex.search(text) is not None


def contains_phone_number(text):
    return phone_no_regex.search(text) is not None


def get_sbert_model(language):
    if language == 'pt':
        return portuguese_sbert
    elif language in ['hi', 'ml', 'mr', 'ta', 'te', 'bn', 'hi-Latn']:
        return indian_sbert
    elif language == 'en':
        return english_sbert
    else:
        return None


def vcosine(u, v):
    return abs(1 - distance.cdist(u, v, 'cosine'))


def get_sbert_embedding(text, language):
    model = get_sbert_model(language)
    if isinstance(text, list) or isinstance(text, tuple):
        return model.encode(text)
    else:
        return model.encode([text])


def get_laser_embedding(text, lang):
    if isinstance(text, list) or isinstance(text, tuple):
        return laser.embed_sentences(text, lang=lang)
    else:
        return laser.embed_sentences([text], lang=lang)


def get_fuzzy_similarity_score(a, b):
    return fuzz.partial_ratio(a, b) / 100