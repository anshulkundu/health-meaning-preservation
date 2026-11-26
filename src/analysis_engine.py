"""Transparent heuristic analysis for health-instruction sentence pairs.
This is a research demo, not a clinically validated meaning-preservation model.
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NEGATION = {
    'en': {'no','not','never','without','dont',"don't","doesn't","shouldn't","mustn't","cannot","can't"},
    'hi': {'नहीं','मत','बिना','न','कभी'}
}
TIME = {
    'en': {'daily','day','days','weekly','week','monthly','month','morning','evening','night','before','after','until','every','hour','hours'},
    'hi': {'दिन','रोज','प्रतिदिन','सुबह','शाम','रात','पहले','बाद','घंटे','हफ्ते','महीने'}
}
WARNING = {
    'en': {'warning','severe','serious','emergency','urgent','danger','avoid','stop','seek','doctor','hospital','immediately'},
    'hi': {'चेतावनी','गंभीर','आपातकाल','तुरंत','खतरा','बचें','रोकें','डॉक्टर','अस्पताल'}
}
UNITS = {'mg','mcg','g','kg','ml','l','tablet','tablets','capsule','capsules','滴'}

def _tokens(text):
    return re.findall(r"[\w]+(?:['’][\w]+)?", str(text).lower(), flags=re.UNICODE)

def _terms(text, vocab, lang='en'):
    lowered = str(text).lower()
    if lang == 'hi':
        # Devanagari combining marks are not consistently captured by \w; use phrase matching.
        return sorted(term for term in vocab if term in lowered)
    toks = set(_tokens(text))
    return sorted(toks.intersection(vocab))

def analyze_pair(original: str, comparison: str, language='English'):
    lang = 'hi' if str(language).lower().startswith('h') else 'en'
    # Word + character n-grams make this baseline somewhat more tolerant of inflections/spacing.
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=1)
    matrix = vectorizer.fit_transform([original, comparison])
    similarity = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    score = round(similarity * 100, 1)

    a, b = _tokens(original), _tokens(comparison)
    flags = []
    na, nb = _terms(original, NEGATION[lang], lang), _terms(comparison, NEGATION[lang], lang)
    if na != nb:
        flags.append({'type':'Negation change','severity':'High','detail':f'Original: {na or "none"}; comparison: {nb or "none"}.'})
    nums_a = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|ml|l|%|hours?|days?|tablets?)?', original.lower())
    nums_b = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|ml|l|%|hours?|days?|tablets?)?', comparison.lower())
    if nums_a != nums_b:
        flags.append({'type':'Number / dosage change','severity':'High','detail':f'Original quantities: {nums_a or "none"}; comparison quantities: {nums_b or "none"}.'})
    ta, tb = _terms(original, TIME[lang], lang), _terms(comparison, TIME[lang], lang)
    if ta != tb:
        flags.append({'type':'Timing change','severity':'High','detail':f'Original timing terms: {ta or "none"}; comparison timing terms: {tb or "none"}.'})
    wa, wb = _terms(original, WARNING[lang], lang), _terms(comparison, WARNING[lang], lang)
    if wa != wb:
        flags.append({'type':'Warning / severity wording change','severity':'Medium','detail':f'Original warning terms: {wa or "none"}; comparison warning terms: {wb or "none"}.'})
    # Surface lexical edits for human inspection; these are not semantic explanations.
    stop = {'the','a','an','and','or','to','of','in','on','with','for','your','this','that','is','are','be','के','का','की','और','में','से'}
    removed = sorted(set(a) - set(b) - stop)
    added = sorted(set(b) - set(a) - stop)
    return {'similarity_score':score, 'flags':flags, 'removed_terms':removed[:30], 'added_terms':added[:30],
            'interpretation':'Potential critical change detected; manual review required.' if flags else 'No configured critical-change pattern detected; manual review is still required.'}
