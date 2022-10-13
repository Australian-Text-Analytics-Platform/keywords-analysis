import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pandas._testing import assert_frame_equal
from tqdm import tqdm

def collapse_corpus_by_source(df):
    '''
    Takes a corpus df with title, body & source columns, one article per row
    Returns a df with n_source rows, where 'text' contains the union of body and title for
    all articles from that source
    '''
    corpusdict = {}
    for source in df.source.unique().tolist():
        body = df[df['source'] == source]['text'].to_list()
        title = df[df['source'] == source]['text_name'].to_list()
        bodystring = ' '.join([str(elem) for elem in body])
        titlestring = ' '.join([str(elem) for elem in title])
        corpusdict[source] = bodystring + " " + titlestring
    df_counting = pd.DataFrame.from_dict(corpusdict, orient='index', columns=['text'])
    df_counting['source'] = df_counting.index
    
    return df_counting


def count_words(df):
    '''
    Takes a df with one row per source, with cols 'source' and 'text'
    Returns a df of word counts from each source in the corpus
    '''
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    corpustokencounts = vectorizer.fit_transform(tqdm(df['text'].values,
                                                 total=len(df), 
                                                 desc='Step 1/3', 
                                                 leave=True))
    wordcount_df = pd.DataFrame(data=np.transpose(corpustokencounts.toarray()), columns=df.source.tolist())
    wordcount_df['word'] = vectorizer.get_feature_names_out()
    
    return wordcount_df


def get_totals(df):
    '''
    Takes a word count df with counts for each source plus 'word'
    Returns df with sum by source plus dict total_by_source & int total_words_in_corpus
    '''
    total_by_source = df.loc[:, df.columns != "word"].sum(axis=0).to_dict()
    total_word_used = df.loc[:, df.columns != "word"].sum(axis=1)
    df['total_word_used'] = total_word_used # total word used (per word)
    total_words_in_corpus = sum(total_word_used)
    
    return (df, total_by_source, total_words_in_corpus)


def single_source_ln(source_wc, expected_wc_source):
    if (source_wc == 0) or (expected_wc_source == 0):
        return 0
    else:
        # unlike the excel spreadsheet put the 2x multiplication here
        return 2 * source_wc * np.log(source_wc/expected_wc_source)


def get_percent_diff(normalised_wc_source, normalised_restofcorpus_wc, diff_zero_freq_adjustment):
    if normalised_restofcorpus_wc == 0:
        divideby = diff_zero_freq_adjustment
    else:
        divideby = normalised_restofcorpus_wc
        
    return 100 * (normalised_wc_source - normalised_restofcorpus_wc)/divideby


def log2_ratio(normalised_freq_source, normalised_freq_rest_of_corpus, total_words_source1, total_words_rest_of_corpus):
    # constant as per spreadsheet
    log_ratio_zero_freq_adjustment = 0.5
    numerator = normalised_freq_source if normalised_freq_source != 0 else log_ratio_zero_freq_adjustment/total_words_source1
    denominator = normalised_freq_rest_of_corpus if normalised_freq_rest_of_corpus != 0 else log_ratio_zero_freq_adjustment/total_words_rest_of_corpus
    
    return np.log2(numerator/denominator)


def odds_ratio(source_wc, rest_of_corpus_wc, total_words_source, total_words_rest_of_corpus):
    numerator = source_wc/(total_words_source-source_wc)
    denominator = rest_of_corpus_wc/(total_words_rest_of_corpus-rest_of_corpus_wc)
    if denominator == 0:
        return np.nan
    else:
        return numerator/denominator


def relative_risk(normalised_wc, normalised_restofcorpus_wc):
    if normalised_restofcorpus_wc == 0:
        return np.nan
    else:
        return normalised_wc/normalised_restofcorpus_wc


def two_corpus_compare(df, total_by_source, total_words_in_corpus):
    '''
    Compare two corpora, as per the 2 corpus Lancaster example
    Works on two corpora
    '''
    outdf = df.copy()
    sources = outdf.columns.difference(['word', 'total_word_used'])
    # comparing two, so df=1
    degrees_of_freedom = 1
    diff_zero_freq_adjustment = 1E-18
    
    for source in tqdm(sources, 
                       total=len(sources),
                       desc='Step 2/3',
                       leave=True):
        # TODO do this with apply for each source, and then merge the tables together adding a "_{source}" 
        # #suffix to each column name. All of the "_"+source in this code get in the way of its readability. 
        # It makes it uncomfortable for me to try check correctness.
        
        # word count of the rest of the corpus = 
        # diff between total words in the corpus minus total words in each source
        wc_restofcorpus = total_words_in_corpus - total_by_source[source] 
        
        # expected word count for each source =
        # total words in each source * total word used (per word) / total words in the corpus
        outdf['expected_wc_'+source] = total_by_source[source] * outdf['total_word_used']/ total_words_in_corpus
        
        # expected word count for the rest of the corpus (excluding the above source) =
        # word count of the rest of the corpus * total word used (per word) / total words in the corpus 
        outdf['expected_restofcorpus_wc_'+source] = wc_restofcorpus * outdf['total_word_used']/total_words_in_corpus
        
        # normalised word count for each source =
        # total words for each word in the source / total words in that source
        outdf['normalised_wc_'+source] = outdf[source]/total_by_source[source]
        
        # normalised word count for the rest of the corpus (excluding the above source) =
        # total words for each word in the rest of the corpus / total words in the rest of the corpus
        outdf['normalised_restofcorpus_wc_'+source] = (outdf['total_word_used'] - outdf[source])/wc_restofcorpus
        
        # determine if the word is overused in that source = 
        # True if normalised word count for in the source > normalised word count for the rest of the corpus
        outdf['overuse_'+source] = outdf['normalised_wc_'+source] > outdf['normalised_restofcorpus_wc_'+source]
        
        # log-likelihood calculation per-source vs rest of corpus
        # calculate outdf['log_likelihood_'+source] = 
        # 2 * x[source] * np.log(x[source]/x['expected_wc_' + source])
        tqdm.pandas(desc='Step 2.1',leave=False)
        tmparray = outdf.apply(lambda x: single_source_ln(x[source], x['expected_wc_' + source]), axis=1).array
        # add 2 * (x['total_word_used'] - x[source]) * np.log((x['total_word_used'] - x[source])/x['expected_restofcorpus_wc_' + source])
        tqdm.pandas(desc='Step 2.2',leave=False)
        tmparray += outdf.apply(lambda x: single_source_ln((x['total_word_used'] - x[source]), x['expected_restofcorpus_wc_' + source]), axis=1).array
        outdf['log_likelihood_'+source] = tmparray
        
        # %diff calculation per-source
        # calculate outdf['percent_diff_'+source] =
        # 100 * (x['normalised_wc_'+source] - x['normalised_restofcorpus_wc_'+source] / x['normalised_restofcorpus_wc_'+source] or 1E-18
        tqdm.pandas(desc='Step 2.3',leave=False)
        outdf['percent_diff_'+source] = outdf.apply(lambda x: get_percent_diff(x['normalised_wc_'+source],x['normalised_restofcorpus_wc_'+source], diff_zero_freq_adjustment), axis=1)
        
        # bayes_factor_bic calculation per-source
        # calculate outdf['bayes_factor_bic_'+source] = 
        # outdf['log_likelihood_'+source] - (degrees_of_freedom * np.log(total_words_in_corpus))
        outdf['bayes_factor_bic_'+source] =  outdf['log_likelihood_'+source] - (degrees_of_freedom * np.log(total_words_in_corpus))
        
        # Effect Size for Log Likelihood (ELL) calculation per-source
        # calculate outdf['ell_'+source] = 
        # outdf['log_likelihood_'+source]/(total_words_in_corpus * np.log(outdf.filter(regex=str('expected.*' + source )).min(axis=1)))
        outdf['ell_'+source] = outdf['log_likelihood_'+source]/(total_words_in_corpus * np.log(outdf.filter(regex=str('expected.*' + source )).min(axis=1)))
        
        # relative_risk calculation per-source
        # calculate outdf['relative_risk_'+source] =
        # x['normalised_wc_'+source]/x['normalised_restofcorpus_wc_'+source] OR np.nan
        tqdm.pandas(desc='Step 2.4',leave=False)
        outdf['relative_risk_'+source] = outdf.apply(lambda x: relative_risk(x['normalised_wc_'+source], x['normalised_restofcorpus_wc_'+source]), axis=1)
        
        # log_ratio calculation per-source
        # calculate outdf['log_ratio_' + source] = 
        # np.log2(numerator/denominator)
        # numerator = normalised_freq_source OR 0.5/total_words_source1
        # denominator = normalised_freq_rest_of_corpus OR 0.5/total_words_rest_of_corpus        
        tqdm.pandas(desc='Step 2.5',leave=False)
        outdf['log_ratio_' + source] = outdf.apply(lambda x: log2_ratio(x['normalised_wc_'+source], x['normalised_restofcorpus_wc_'+source], total_by_source[source], wc_restofcorpus), axis=1)
        
        # odds_ratio calculation per-source
        # calculate outdf['odds_ratio_' + source] =
        # numerator/denominator OR np.nan
        # numerator = source_wc/(total_words_source-source_wc)
        # denominator = rest_of_corpus_wc/(total_words_rest_of_corpus-rest_of_corpus_wc)
        tqdm.pandas(desc='Step 2.6',leave=False)
        outdf['odds_ratio_' + source] = outdf.apply(lambda x: odds_ratio(x[source], (x['total_word_used'] - x[source]), total_by_source[source], wc_restofcorpus), axis=1)
    
    return outdf


def test_twocorpus_compare(projectroot):
    # test file has raw data and results
    test2 = pd.read_csv(projectroot/"100_data_raw"/"211008_SigEff_2test.csv")
    # keep only word counts and the word in there
    test2_wc_only = test2[test2.columns[pd.Series(test2.columns).str.startswith('corpus')]].copy()
    test2_wc_only['word'] = test2['word']
    testcountdf, total_by_source_test, total_words_in_corpus_test = get_totals(df=test2_wc_only)
    twocorp_comparison_test = two_corpus_compare(df=testcountdf, total_by_source=total_by_source_test, total_words_in_corpus=total_words_in_corpus_test)
    # the assertion passes :)
    return assert_frame_equal(twocorp_comparison_test[test2.columns.to_list()], test2, check_exact=False, atol=0.1, rtol=0.1)


def n_corpus_compare(df, total_by_source, total_words_in_corpus):
    '''
    Compare multiple corpora, as per the 6 corpus Lancaster example
    Works on 2+ - infinity corpora correctly
    '''
    outdf = df.copy()
    sources = outdf.columns.difference(['word', 'total_word_used'])
    for source in sources:
        outdf[str('expected_wc_'+source)] = total_by_source[source] * outdf['total_word_used'] / total_words_in_corpus

    for source in tqdm(sources,
                       total=len(sources),
                       desc='Step 3/3',
                       leave=True):
        if source == sources[0]:
            # first source
            tqdm.pandas(desc='Step 3.1',leave=False)
            tmparray = outdf.apply(lambda x: single_source_ln(x[source], x[str('expected_wc_' + source)]), axis=1).array
        else:
            sub_step = 2
            leave=False
            if sub_step==len(sources)-1:
                leave=True
            desc = 'Step 3.{}'.format(sub_step)
            tqdm.pandas(desc=desc,leave=leave)
            tmparray += outdf.apply(lambda x: single_source_ln(x[source], x[str('expected_wc_' + source)]), axis=1).array
            sub_step +=1

    outdf['Log Likelihood'] = tmparray
    degrees_of_freedom = len(sources) - 1
    outdf['Bayes Factor BIC'] = outdf['Log Likelihood'] - (degrees_of_freedom * np.log(total_words_in_corpus))
    outdf['ELL'] = outdf['Log Likelihood']/(total_words_in_corpus * np.log(outdf.filter(regex='expected_wc').min(axis=1)))
    
    return outdf


def test_multicorpus_compare(projectroot):
    # test file has raw data and results
    test6 = pd.read_csv(projectroot/"100_data_raw"/"211008_SigEff_6test.csv")
    # keep only word counts and the word in there
    test6_wc_only = test6[test6.columns[pd.Series(test6.columns).str.startswith('corpus')]].copy()
    test6_wc_only['word'] = test6['word']
    testcountdf, total_by_source_test, total_words_in_corpus_test = get_totals(df=test6_wc_only)
    multicorp_comparison_test = n_corpus_compare(df=testcountdf, total_by_source=total_by_source_test, total_words_in_corpus=total_words_in_corpus_test)
    
    return assert_frame_equal(multicorp_comparison_test, test6)