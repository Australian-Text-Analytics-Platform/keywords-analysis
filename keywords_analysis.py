#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:15:24 2022

@author: sjuf9909
"""
# import required packages
import codecs
import hashlib
import io
import os
from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
from pyexcelerate import Workbook

# numpy and pandas: tools for data processing
import pandas as pd
import numpy as np

# matplotlib and seaborn: visualization tools
import matplotlib.pyplot as plt

# Bokeh: interactive plots
from bokeh.io import output_notebook
from bokeh.models import FixedTicker
from bokeh.plotting import figure, show
output_notebook()

# nltk: natural language processing toolkit
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# scikit-learn: machine learning tool
from sklearn.feature_extraction.text import CountVectorizer

# scipy: collection of math algorithms and functions built on the NumPy extension of Python
from scipy.stats import boxcox, ttest_ind, permutation_test

# ipywidgets: tools for interactive browser controls in Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, clear_output, FileLink, HTML

# import other packages
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()
from multicorpus_comparison_functs import collapse_corpus_by_source, count_words, get_totals
from multicorpus_comparison_functs import two_corpus_compare
from multicorpus_comparison_functs import n_corpus_compare


class DownloadFileLink(FileLink):
    '''
    Create link to download files in Jupyter Notebook
    '''
    html_link_str = "<a href='{link}' download={file_name}>{link_text}</a>"

    def __init__(self, path, file_name=None, link_text=None, *args, **kwargs):
        super(DownloadFileLink, self).__init__(path, *args, **kwargs)

        self.file_name = file_name or os.path.split(path)[1]
        self.link_text = link_text or self.file_name

    def _format_path(self):
        from html import escape

        fp = "".join([self.url_prefix, escape(self.path)])
        return "".join(
            [
                self.result_html_prefix,
                self.html_link_str.format(
                    link=fp, file_name=self.file_name, link_text=self.link_text
                ),
                self.result_html_suffix,
            ]
        )


class KeywordsAnalysis():
    '''
    Using word statistics to analyse words in a collection of corpus 
    and identify whether certain words are over or under-represented 
    in a particular corpus compared to their representation in other 
    corpus
    '''
    def __init__(self):
        '''
        Initiate the KeywordsAnalysis
        '''
        # initiate required variables
        self.large_file_size = 1000000
        self.text_df = pd.DataFrame()
        self.freq_df = pd.DataFrame()
        self.corpus_options = []
        self.current_study_corpus = None
        self.current_ref_corpus = None
        self.new_display = True
        
        # CSS styling 
        self.style = self.text_styling()
    
    
    def upload_file_widget(self):
        '''
        Function to upload text files
        '''
        # widget for entering corpus name
        corpus_name = widgets.Text(
            value='',
            placeholder='Enter corpus name...',
            description='Corpus Name:',
            disabled=False,
            style= {'description_width': 'initial'},
            layout = widgets.Layout(margin='0px 20px 0px 0px')
        )
        
        # widget for frequency list checkbox
        freq_list = widgets.Checkbox(
            value=False,
            description='Uploading word frequency list',
            disabled=False,
            indent=False
        )
        
        # widget for file upload
        self.file_uploader = widgets.FileUpload(
            description='Upload your files (txt, csv, xlsx or zip)',
            accept='.txt, .xlsx, .csv, .zip', # accepted file extension
            multiple=True,  # True to accept multiple files
            error='File upload unsuccessful. Please try again!',
            layout = widgets.Layout(width='300px',
                                    margin='5px 0px 0px 0px')
            )
    
        upload_out = widgets.Output()
        warning_out = widgets.Output()
        
        with warning_out:
            self.upload_warning()
        
        # give notification when file is uploaded
        def _cb(change):
            with upload_out:
                if self.file_uploader.value!=():
                    try:
                        # clear output and give notification that file is being uploaded
                        clear_output()
                        
                        # check file size
                        self.check_file_size(self.file_uploader)
                        
                        # reading uploaded files
                        self.process_upload(freq_list.value,
                                            corpus_name.value)
                        
                        # reset upload widget values
                        corpus_name.value = ''
                        corpus_name.placeholder='Enter corpus name...'
                        freq_list.value = False
                        
                        # give notification when uploading is finished
                        self.notify_and_reset_upload_widget()
                    except:
                        print('\033[1m\nFile upload unsuccessful...\033[0m')
                        print("Please ensure to tick the 'Uploading word frequency list' box when uploading a frequency list.")
                        
                # clear saved value in cache and reset counter
                self.file_uploader.value = ()
        
        # observe when file is uploaded and display output
        self.file_uploader.observe(_cb, names='value')
        
        # design widgets layout
        hbox = widgets.HBox([corpus_name, freq_list])
        
        upload_box = widgets.VBox([hbox, 
                                   self.file_uploader, 
                                   warning_out,
                                   upload_out])
        
        return upload_box
    
    
    def upload_warning(self):
        '''
        Function to print warning when uploading texts
        '''
        print('Uploading large files may take a while. Please be patient.')
        print('\033[1mPlease wait and do not press any buttons until the progress bar appears...\033[0m')
    
    
    def check_file_size(self, uploaded_file):
        '''
        Function to check the uploaded file size
        
        Args:
            uploaded_file: the uploaded file containing the text data
        '''
        # check total uploaded file size
        total_file_size = sum([file['size'] for file in uploaded_file.value])
        print('The total size of the upload is {:.2f} MB.'.format(total_file_size/1000000))
        
        # display warning for individual large files (>1MB)
        large_text = [file['name'] for file in uploaded_file.value \
                      if file['size']>self.large_file_size and \
                          file['name'].endswith('.txt')]
        if len(large_text)>0:
            print('The following file(s) are larger than 1MB:', large_text)
        
        
    def process_upload(self, 
                       freq_list: bool = False,
                       corpus_name: str = '',
                       deduplication: bool = True):
        '''
        Pre-process uploaded .txt files into pandas dataframe

        Args:
            freq_list: whether the uploaded file is a word frequenxy list
            corpus_name: the name of the corpus
            deduplication: option to deduplicate text_df by text_id
        '''
        # create an input folder if not already exist
        os.makedirs('input', exist_ok=True)
        
        # get uploaded files
        files = self.get_uploaded_files(self.file_uploader.value)
        
        print('Reading uploaded files...')
        print('This may take a while...')
        # read uploaded files
        all_data = self.read_uploaded_files(files, freq_list, corpus_name)
        
        # remove files and directory once finished
        os.system('rm -r ./input')
    
        # convert them into a pandas dataframe format and add unique id
        temp_df = pd.DataFrame.from_dict(all_data)
        
        if 'source' not in temp_df.columns:
            temp_df['source'] = len(temp_df) * [corpus_name]
        
        if len(temp_df)!=0:
            temp_df = self.hash_gen(temp_df)
            
        self.text_df = pd.concat([self.text_df,temp_df])
    
        # deduplicate the text_df by text_id
        if deduplication:
            self.text_df.drop_duplicates(subset='text_id', keep='first', inplace=True)
    
    
    def freq_list_warning(self):
        '''
        Function to print warning when uploading word frequency list
        '''
        print('\033[1mWord frequency list upload unsuccessful!')
        print('Please upload an excel spreadsheet for your word frequency list!\033[0m')
    
    
    def read_uploaded_files(self, files, freq_list, corpus_name):
        '''
        Function to read uploaded files
        
        Args:
            files: the uplaoded files
            freq_list: whether the uploaded file is a word frequenxy list
            corpus_name: the name of the corpus
        '''
        all_data = []
        # process and upload files
        for file, n in tqdm(files):
            # process text files
            if str(file).lower().endswith('txt'):
                if freq_list:
                    self.freq_list_warning()
                    text_dic=[]
                else:
                    text_dic = self.load_txt(file, n, freq_list)
            # process xlsx or csv files
            else:
                text_dic = self.load_table(file, n, freq_list, corpus_name)
            all_data.extend(text_dic)
        
        return all_data
    
    
    def get_uploaded_files(self, uploaded_files):
        '''
        Extract zip file and add extracted files to the list of files
        
        Args:
            uploaded_files: the uploaded zip file containing the zipped data
        '''
        files = []
        
        # get uploaded files
        for n, file in enumerate(uploaded_files):
            files.append([file.name, n])
            
            # extract zip files (if any)
            if file.name.lower().endswith('zip'):
                self.extract_zip(self.file_uploader.value[n])
                files.pop()
        
        # add extracted files to files
        for file_type in ['*.txt', '*.xlsx', '*.csv']:
            files += [[file, 'unzip'] for file in Path('./input').rglob(file_type) if 'MACOSX' not in str(file)]
        
        return files
    
    
    def extract_zip(self, zip_file):
        '''
        Load zip file
        
        Args:
            zip_file: the file containing the zipped data
        '''
        # read and decode the zip file
        temp = io.BytesIO(zip_file['content'])
        
        # open and extract the zip file
        with ZipFile(temp, 'r') as zip:
            # extract files
            print('Extracting {}...'.format(zip_file['name']))
            zip.extractall('./input/')
        
        # clear up temp
        temp = None
    
    
    def load_txt(self, file, n, freq_list) -> list:
        '''
        Load individual txt file content and return a dictionary object, 
        wrapped in a list so it can be merged with list of pervious file contents.
        
        Args:
            file: the file containing the text data
            n: index of the uploaded file (value='unzip' if the file is extracted form a zip file)
            freq_list: whether the uploaded file is a word frequenxy list
        '''
        # read the unzip text file
        if n=='unzip':
            # read the unzip text file
            with open(file) as f:
                temp = {'text_name': file.name[:-4],
                        'text': f.read()
                }
            
            os.remove(file)
        else:
            file = self.file_uploader.value[n]
            # read and decode uploaded text
            temp = {'text_name': file['name'][:-4],
                    'text': codecs.decode(file['content'], encoding='utf-8', errors='replace')
            }
            
            # check for unknown characters and display warning if any
            unknown_count = temp['text'].count('�')
            if unknown_count>0:
                print('We identified {} unknown character(s) in the following text: {}'.format(unknown_count, file['name'][:-4]))
        
        return [temp]


    def load_table(self, 
                   file, 
                   n: int, 
                   freq_list: bool = False, 
                   corpus_name: str = '') -> list:
        '''
        Load csv or xlsx file
        
        Args:
            file: the file containing the excel or csv data
            n: index of the uploaded file (value='unzip' if the file is extracted form a zip file)
            freq_list: whether the uploaded file is a word frequenxy list
            corpus_name: the name of the corpus
        '''
        if n!='unzip':
            file = io.BytesIO(self.file_uploader.value[n]['content'])
        
        # read the file based on the file format
        try:
            temp_df = pd.read_csv(file)
        except:
            temp_df = pd.read_excel(file)
            
        if freq_list:
            # load word frequency list
            temp = self.load_freq_list(temp_df, corpus_name, n)
        else:
            # load text files
            temp = self.load_table_text(temp_df, n)
            
        return temp
    
    
    def load_freq_list(self, df, corpus_name, n):
        '''
        Function to load word frequency list
        
        Args:
            df: the dataframe containing the word frequency list
            corpus_name: the name of the corpus
            n: index of the uploaded file (value='unzip' if the file is extracted form a zip file)
        '''
        # check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('word' not in df.columns) or ('freq' not in df.columns):
            print('\033[1mFrequency list upload unsuccessful...\033[0m')
            print('File {} does not contain the required header "word" and "freq"\n'.format(self.file_uploader.value[n]['name']))
        else:
            # rename the columns to word and corpus name
            df.rename(columns={df.columns[0]: 'word',
                               df.columns[1]: corpus_name}, 
                      inplace=True)
            
            # lower case the words
            df['word'] = df['word'].apply(lambda x: str(x).lower())
            
            # add data to the word frequency dataframe
            self.freq_df = self.merge_dataframes(self.freq_df, df)
            
            print('\033[1mUploading word requency list successful!\033[0m')
            
        return []
    
    
    def load_table_text(self, df, n):
        '''
        Function to load table text
        
        Args:
            df: the dataframe containing the word frequency list
            n: index of the uploaded file (value='unzip' if the file is extracted form a zip file)
        '''
        # check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('text' not in df.columns) or ('text_name' not in df.columns):
            print('File {} does not contain the required header "text" and "text_name"'.format(self.file_uploader.value[n]['name']))
            temp = []
        
        # return a list of dict objects
        temp = df[['text_name', 'text', 'source']].to_dict(orient='index').values()
        
        return temp
    
    
    def merge_dataframes(self, df1, df2):
        '''
        Function to merge two dataframes
        
        Args:
            df1: the first dataframe 
            df2: the second dataframe 
        '''
        # add to self.freq_df dataframe
        if len(df1)==0:
            df1 = df2
        else:
            df1 = df1.merge(df2, how='outer', left_on='word', right_on='word')
            df1.fillna(0, inplace=True)
        
        return df1
    
    
    def hash_gen(self, temp_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create column text_id by md5 hash of the text in text_df
        
        Args:
            temp_df: the temporary pandas dataframe containing the text data
        '''
        temp_df['text_id'] = temp_df['text'].apply(
            lambda t: hashlib.shake_256(t.encode('utf-8')).hexdigest(5))
        
        return temp_df
        
        
    def notify_and_reset_upload_widget(self):
        '''
        Function to give notification when upload is finished
        '''
        # give notification when uploading is finished
        print('Finished uploading files.')
        print('{} text documents are loaded.'.format(self.text_df.shape[0]))
        print('\nYou can now upload your next corpus, or continue to the next step')
        
    
    def calculate_word_statistics(self):
        '''
        Function to calculate word statistics in a collection of corpus
        The statistics include normalised word count, log-likelihood, 
        Bayes factor BIC, effect size for log-likelihood (ELL), 
        relative risk, log ratio and odds ratio. For more information, 
        please visit this website: https://ucrel.lancs.ac.uk/llwizard.html
        '''
        # define corpus list
        temp_wordcount_df = pd.DataFrame()
        if len(self.text_df)!=0:
            self.corpus_options += list(set(self.text_df.source))
            self.corpus_options = list(set(self.corpus_options))
            
            # collate all texts based on source and 
            # use CountVectorizer to count the number of words in each source
            temp_wordcount_df = collapse_corpus_by_source(df=self.text_df)
            temp_wordcount_df = count_words(temp_wordcount_df)
        
        # if word frequency list is uploaded
        if len(self.freq_df)!=0:
            self.corpus_options += list(self.freq_df.columns.difference(['word']))
            
            temp_wordcount_df = self.merge_dataframes(temp_wordcount_df, self.freq_df)
            temp_wordcount_df = pd.concat([temp_wordcount_df[temp_wordcount_df.columns.difference(['word'])], 
                                           temp_wordcount_df['word']], axis=1)
        
        # get total word counts based on source and overall in the corpus
        self.wordcount_df, self.total_by_source, self.total_words_in_corpus = get_totals(temp_wordcount_df)
        
        # multi-corpora analysis
        self.multicorp_comparison = n_corpus_compare(self.wordcount_df, 
                                                     self.total_by_source, 
                                                     self.total_words_in_corpus)
        
        self.all_words = self.multicorp_comparison.word.to_list()
        
        
    def visualize_stats(self, 
                        df: pd.DataFrame, 
                        index: int, 
                        inc_chart: list, 
                        title: str, 
                        right_padding: float,
                        multi: bool = False):
        '''
        Function to visualize the calculated statistics onto a line chart

        Args:
            df: the pandas dataframe containing the selected data 
            index: the index of the first word 
            inc_chart: the list of statistics to be included in the chart 
            title: the title of the chart 
            right_padding: padding for the legend box on the right hand side
            multi: whether the chart is for multi-corpora analysis or not
        '''
        # define line color and line dash options
        line_colors = {0: 'tomato', 1: 'indigo',
                       2: 'coral', 3: 'firebrick',
                       4: '#a240a2', 5: '#2F2F2F',
                       6: 'olivedrab', 7: 'green'}
        line_dashes = {0:'solid', 1: 'dotted', 
                       2: 'dashed', 3: 'dotdash',
                       4: 'dashdot', 5: 'dotted', 
                       6: 'dashed', 7: 'dotdash'}
        
        # define the data
        data = df.iloc[index:index+30,inc_chart]
        
        # create the line chart
        x = list(range(0, len(data)))

        p = figure(title=title,
                   background_fill_color="#fafafa",
                   plot_width=950,#900, 
                   plot_height=400)
        
        for n, column in enumerate(data.columns.to_list()):
            p.line(x, data[column], legend_label=column,
                   line_color=line_colors[n], 
                   line_dash=line_dashes[n],
                   line_width=2)
        
        # define the x-ticks
        word_list = data.index.to_list()
        word_list = [word[:20] for word in word_list]
        p.xaxis.ticker = FixedTicker(ticks=list(range(0, len(x))))
        p.xaxis.major_label_overrides = dict(zip(x, word_list))
        p.xaxis.major_label_orientation = 45
        p.xaxis.axis_line_width = 5
        
        # define other chart parameters
        p.xaxis.axis_label_text_font_size = '15px'
        p.yaxis.axis_label_text_font_size = '15px'
        p.xaxis.major_label_text_font_size = '14px'
        p.yaxis.major_label_text_font_size = '14px'
        p.xaxis.axis_label = 'word tokens, index {} to {}'.format(str(index), str(index+30))
        p.yaxis.axis_label = 'statistic values'
        p.x_range.range_padding = right_padding
        p.x_range.start=-1
        p.legend.click_policy = 'hide'
        show(p)
        
    
    def create_graphs(self, 
                      viz_df: pd.DataFrame, 
                      index: int, 
                      inc_corpus1: str, 
                      inc_corpus2: str, 
                      inc_charts: list, 
                      options: dict(), 
                      sort_value: str, 
                      right_padding: float, 
                      multi: bool):
        '''
        Function to generate line charts based on selected parameters

        Args:
            viz_df: the pandas dataframe containing the selected data 
            index: the index of the first word 
            inc_corpus: the list of corpus to be included in the chart 
            inc_charts: the list of statistics to be included in the chart 
            options: the dictionary containing the statistic options to display 
            sort_value: how to sort the statistics 
            right_padding: padding for the legend box on the right hand side 
            multi: whether the chart is for multi-corpora analysis or not 
        '''
        inc_chart = [options[chart][0] for chart in inc_charts]
        
        if multi:
            fig_title = '{}, sorted by: {}'.format('multi-corpus', sort_value)
        else:
            fig_title = 'Study corpus: {}; reference corpus: {}; sorted by: {}'.format(inc_corpus1, 
                                                                                       inc_corpus2,
                                                                                       sort_value)
            
        if sort_value!='alphabetically':
            viz_df = viz_df.sort_values(by=options[sort_value][1], 
                                ascending=False)
        
        self.visualize_stats(viz_df, 
                             index, 
                             inc_chart, 
                             fig_title, 
                             right_padding,
                             multi)
        
        
    def analyse_stats(
            self, 
            right_padding: float,
            multi: bool = False
            ):
        '''
        Function to generate widgets for analysing calculated statistics

        Args:
            range_padding: padding for the legend box on the right hand side 
            multi: whether the chart is for multi-corpora analysis or not 
        '''
        # define variables
        self.figs = []
        
        # whether to do pairwise or multi-corpora analysis
        if multi:
            options = {'log-likelihood':[-3, 'Log Likelihood'],
                       'bayes factor BIC':[-2, 'Bayes Factor BIC'],
                       'ELL':[-1, 'ELL']}
            select_chart_value = list(options.keys())
        else:
            options = {'normalised word count (study corpus)':[-10,'normalised_study_corpus_wc'],
                       'normalised word count (reference corpus)':[-9,'normalised_reference_corpus_wc'],
                       'log-likelihood':[-7,'log_likelihood'],
                       'percent-diff':[-6,'percent_diff'],
                       'bayes factor BIC':[-5,'bayes_factor_bic'],
                       'ELL':[-4,'ell'],
                       'relative risk':[-3,'relative_risk'],
                       'log ratio':[-2,'log_ratio'],
                       'odds ratio':[-1,'odds_ratio']}
            select_chart_value = list(options.keys())[2:]
        
        # widget to select study corpus to include in the analysis
        enter_corpus, select_corpus = self.select_options('<b>Select study corpus:</b>',
                                                          self.corpus_options,
                                                          self.corpus_options[0], 
                                                          '150px')
        
        # widget to select reference corpus to include in the analysis
        enter_ref_corpus, select_ref_corpus = self.select_options('<b>Select reference corpus:</b>',
                                                                  self.corpus_options+['rest of corpus'],
                                                                  'rest of corpus', 
                                                                  '150px')
            
        # widget to select statistics to be included in the line chart
        enter_chart, select_chart = self.select_multiple_options('<b>Select statistic(s) to display):</b>',
                                                               list(options.keys()),
                                                               ['log-likelihood'], #select_chart_value,
                                                               '230px')
        
        # widgets to select how to sort the data
        sort_options = ['alphabetically'] + list(options.keys())
        enter_sort, select_sort = self.select_options(instruction='<b>Sorted by:</b>',
                                                      options=sort_options,
                                                      value='log-likelihood')
        
        # widgets to select positive/negative/all words
        keyword_options = ['positive (overuse)', 'negative (underuse)', 'all keywords']
        enter_keyword, select_keyword = self.select_options(instruction='<b>Keywords to display:</b>',
                                                      options=keyword_options,
                                                      value='positive (overuse)')
        
        # widget to display analysis
        display_button, display_out = self.click_button_widget(desc='Display chart',
                                                       margin='10px 32px 0px 2px',
                                                       width='180px')
        
        # function to define what happens when the display button is clicked
        def on_display_button_clicked(_):
            with save_out:
                # clear output and get word index
                clear_output()
            with display_out:
                # clear output and get word index
                clear_output()
                index=0
                display_index.value=0
                self.new_display=True
                
                if select_corpus.value!=select_ref_corpus.value:
                    if multi:
                        viz_df = self.multicorp_comparison.copy()
                    else:
                        if (self.current_study_corpus!=select_corpus.value) or \
                            (self.current_ref_corpus!=select_ref_corpus.value):
                            self.pairwise_compare = two_corpus_compare(self.wordcount_df, 
                                                                   select_corpus.value,
                                                                   select_ref_corpus.value,
                                                                   self.total_by_source, 
                                                                   self.total_words_in_corpus)
                            self.current_study_corpus = select_corpus.value
                            self.current_ref_corpus = select_ref_corpus.value
                    
                        viz_df = self.pairwise_compare.copy()
                        
                        # set the keywords to display (positive/negative/all)
                        if select_keyword.value=='positive (overuse)':
                            viz_df = viz_df[viz_df['percent_diff']>0]
                        elif select_keyword.value=='negative (underuse)':
                            viz_df = viz_df[viz_df['percent_diff']<0]
                    
                    # set the words as the index of the dataframe
                    viz_df.set_index('word', inplace=True)
                    
                    # display updated charts
                    self.create_graphs(viz_df, 
                                       index, 
                                       select_corpus.value, 
                                       select_ref_corpus.value,
                                       select_chart.value, 
                                       options, 
                                       select_sort.value, 
                                       right_padding, 
                                       multi)
                else:
                    print('Please select a different study and reference corpora!')
                
        # link the display button with the function
        display_button.on_click(on_display_button_clicked)
        
        enter_index, display_index = self.select_n_widget('<b>Select index:</b>', 
                                                          value=0,
                                                          min_value=0,
                                                          max_value=len(self.all_words[:-20]))
        
        # update charts when the slider is moved
        def _cb(change):
            with display_out:
                if not(display_index.value==0 and self.new_display==False):
                    # clear output and get word index
                    clear_output()
                    index= display_index.value#self.all_words.index(display_index.value)
                    
                    if multi:
                        viz_df = self.multicorp_comparison.copy()
                    else:
                        viz_df = self.pairwise_compare.copy()
                        
                        # set the keywords to display (positive/negative/all)
                        if select_keyword.value=='positive (overuse)':
                            viz_df = viz_df[viz_df['percent_diff']>0]
                        elif select_keyword.value=='negative (underuse)':
                            viz_df = viz_df[viz_df['percent_diff']<0]
                    
                    # set the words as the index of the dataframe
                    viz_df.set_index('word', inplace=True)
                    
                    # display updated charts
                    self.create_graphs(viz_df, 
                                       index, 
                                       select_corpus.value, 
                                       select_ref_corpus.value,
                                       select_chart.value, 
                                       options, 
                                       select_sort.value, 
                                       right_padding, 
                                       multi)
                    self.new_display=False
                    
        # widget to sav analysis
        save_button, save_out = self.click_button_widget(desc='Save data to excel',
                                                       margin='10px 32px 0px 2px',
                                                       width='180px')
        
        # function to define what happens when the display button is clicked
        def on_save_button_clicked(_):
            with save_out:
                # clear output and get word index
                clear_output()
                
                # specify the saving parameters
                output_dir = './output/'
                
                try:
                    if multi:
                        df = self.multicorp_comparison
                        file_name = 'multi_corpus_analysis.xlsx'
                        sheet_name = 'multi-corpus-analysis'
                    else:
                        df = self.pairwise_compare
                        sheet_name = 'pairwise_analysis'
                        file_name = 'study_{}_ref_{}.xlsx'.format(select_corpus.value,
                                                              select_ref_corpus.value)
                        
                        # drop excluded corpus from the table
                        if select_ref_corpus.value!='rest of corpus':
                            included_corpus = np.array([select_corpus.value,select_ref_corpus.value])
                            excluded_corpus = np.setdiff1d(self.text_df.source.unique(),included_corpus)
                            df = df.drop(excluded_corpus, axis=1)
                                
                except:
                    print('Please generate and display a chart first before saving!')
                
                # select the number of rows to display
                display_n = 5
                
                # save and display the first n rows
                self.save_analysis(df, output_dir, file_name, sheet_name, display_n)
                
        # link the display button with the function
        save_button.on_click(on_save_button_clicked)
        
        # observe when selection slider is moved
        display_index.observe(_cb, names='value')
        
        # displaying inputs, buttons and their outputs
        vbox1 = widgets.VBox([enter_corpus,
                              select_corpus,
                              enter_ref_corpus, 
                              select_ref_corpus], 
                             layout = widgets.Layout(width='180px'))#, height='150px'))
        
        vbox2 = widgets.VBox([enter_chart, 
                              select_chart], 
                             layout = widgets.Layout(width='260px'))#, height='200px'))
        
        vbox4 = widgets.VBox([enter_index, 
                              display_index,
                              display_button, 
                              save_button], 
                             layout = widgets.Layout(width='220px', height='180px'))
        
        # exclude corpus selection for multi-corpora analysis
        if multi:
            vbox3 = widgets.VBox([enter_sort, 
                                  select_sort], 
                                 layout = widgets.Layout(width='210px'))#, height='200px'))
            hbox1 = widgets.HBox([vbox2, vbox3, vbox4])
        else:
            vbox3 = widgets.VBox([enter_sort, 
                                  select_sort,
                                  enter_keyword, 
                                  select_keyword], 
                                 layout = widgets.Layout(width='210px'))#, height='200px'))
            hbox1 = widgets.HBox([vbox1, vbox2, vbox3, vbox4])
            
        vbox = widgets.VBox([hbox1, display_out, save_out])
        
        return vbox
    
    
    def save_analysis(self, 
                      df: pd.DataFrame, 
                      output_dir: str,
                      file_name: str, 
                      sheet_name: str,
                      display_n: int = 5):
        '''
        Function to save analysis into an excel spreadsheet and download to local computer

        Args:
            df: the pandas dataframe containing the selected data 
            output_dir: the name of the output directory.
            file_name: the name of the saved file 
            sheet_name: the sheet name of the excel spreadsheet 
            display_n: the number of rows to display on the notebook 
        '''
        # create output folder if not already exist
        os.makedirs('output', exist_ok=True)
        
        # display the first n rows in html format
        df_html = df.head(display_n).to_html(escape=False)
        
        # apply css styling
        df_html = self.style+'<div class="dataframe-div">'+df_html+"\n</div>"
        
        # display analysis
        display(HTML(df_html))
        
        print('Saving in progress...')
        # save analysis into an Excel spreadsheet
        values = [df.columns] + list(df.values)
        wb = Workbook()
        wb.new_sheet(sheet_name, data=values)
        wb.save(output_dir + file_name)
        
        # download the file onto your computer
        print('Saving is complete.')
        print('Click below to download:')
        display(DownloadFileLink(output_dir + file_name, file_name))
        
        
    def text_length(self):
        '''
        Function to calculate text length
        '''
        # function to calculate the length of each text in the dataset
        def text_len(text):
            return len(word_tokenize(text))
        
        # apply the above function to the text
        tqdm.pandas()
        self.text_df['text_len'] = self.text_df['text'].progress_apply(text_len)
        
    
    def word_count(self, word: str):
        '''
        Function to calculate word count in the text
        
        Args:
            word: the word to be counted 
        '''
        # use CountVectorizer and the 'word' as the patern'
        vectorizer = CountVectorizer(token_pattern=word)
        
        # calculate the word count for the selected 'word'
        corpustokencounts = vectorizer.fit_transform(tqdm(self.text_df['text'].values,
                                                 total=len(self.text_df), 
                                                 desc='', 
                                                 leave=True))
        
        # insert into text_df and normalised (per 1,000 words)
        for n, word in enumerate(vectorizer.get_feature_names_out()):
            self.text_df[word] = corpustokencounts.toarray()[:,n]
            self.text_df[word+'_per_1000_words'] = self.text_df[word]/self.text_df['text_len']*1000
            
            
    def check_length(self, ds: pd.Series):
        '''
        Function to check the length of the data
        
        Args:
            ds:  the data in pandas Series format
        '''
        if len(ds)==1:
            ds = pd.concat([ds,pd.Series([0])])
        
        return ds
    
    
    def x_and_y(self, 
                word: str, 
                source_1: str, 
                source_2: str,
                d_trans: float):
        '''
        Function to calculate x and y for histogram and statistical analysis
        
        Args:
            word:  the 'word' to analyse
            source_1: the first corpus to be compared
            source_2: the second corpus to be compared
            d_trans: data transformation to perform
        '''
        x = self.text_df[self.text_df['source']==source_1][word+'_per_1000_words']
        y = self.text_df[self.text_df['source']==source_2][word+'_per_1000_words']
        
        # exclude zero values (articles where the word did not appear in them)
        x = x[x!=0]
        y = y[y!=0]
        
        x = boxcox(x, d_trans)
        y = boxcox(y, d_trans)
        
        x = self.check_length(x)
        y = self.check_length(y)
        
        return x, y
    
    
    def welch_t_test(self, x, y, alt: str = 'two-sided'):
        '''
        Function to calculate Welch t-test
        
        Args:
            x: the data (samples) from corpus 1 
            y: the data (samples) from corpus 2 
            alt:  the alternative hypothesis, which could include:
                1. ‘two-sided’: the means of the distributions underlying the samples are unequal
                2. ‘less’: the mean of the distribution underlying the first sample is less than the mean of the distribution underlying the second sample
                3. ‘greater’: the mean of the distribution underlying the first sample is greater than the mean of the distribution underlying the second sample
        '''
        # use scipy's ttest_ind to perform welch t-test
        welch = ttest_ind(x, y, equal_var=False, alternative=alt, random_state=42)
        
        return welch.statistic, welch.pvalue
    
    
    def fisher_permutation_test(self, 
                                x, 
                                y, 
                                alt: str = 'two-sided',
                                n_resamples: int = 9999):
        '''
        Function to calculate Fisher permutation test
        
        Args:
            x: the data (samples) from corpus 1 
            y: the data (samples) from corpus 2 
            alt:  the alternative hypothesis, which could include:
                1. ‘two-sided’: the means of the distributions underlying the samples are unequal
                2. ‘less’: the mean of the distribution underlying the first sample is less than the mean of the distribution underlying the second sample
                3. ‘greater’: the mean of the distribution underlying the first sample is greater than the mean of the distribution underlying the second sample
            n_resamples: number of random permutations (resamples) used to approximate the null distribution
        '''
        # function to calculate mean difference
        def statistic(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        
        # use scipy's permutation_test to perform fisher permutation test
        fisher = permutation_test((x, y), statistic, vectorized=True, 
                       n_resamples=9999, alternative=alt, random_state=42)
        
        return fisher.statistic, fisher.pvalue
    
    
    def plot_histogram(self,
                       x,
                       y,
                       word: str,
                       source_1: str, 
                       source_2: str,
                       d_trans: str):
        '''
        Function to plot histogram of word frequency distribution
        
        Args:
            x: the data (samples) from corpus 1 
            y: the data (samples) from corpus 2 
            word:  the 'word' to analyse
            source_1: the first corpus to be compared
            source_2: the second corpus to be compared
            d_trans: the type of data transformation performed on the data
        '''
        # use matplotlib to plot histogram of data distribution
        plt.hist(x, bins='auto', alpha=0.5, label=source_1)
        plt.hist(y, bins='auto', alpha=0.5, label=source_2)
        plt.title("Word '{}' distribution in {} vs. {}".format(word, source_1, source_2))
        plt.ylabel('number of articles')
        plt.xlabel('{}(word frequency per 1,000 words)'.format(d_trans))
        plt.legend(bbox_to_anchor=(1.5, 1))
        plt.show()
    
    
    def word_usage_analysis(self):
        '''
        Function to generate widgets for analysing word using Welch t-test or Fisher permutation test
        '''
        # calculate text length
        self.text_length()
        
        # widget to display word instruction
        word_instruction = widgets.HTML(
            value='Enter the word you wish to analyse',
            placeholder='',
            description=''
            )
        
        # widget to enter the word
        enter_word, word = self.enter_word_widget()
        
        # widget to display corpora instruction
        corpora_instruction = widgets.HTML(
            value='Enter the name of the two corpora you wish to compare',
            placeholder='',
            description=''
            )
        
        # widgets to select first source
        enter_source_1, source_1 = self.select_text_widget(value='Corpus 1:', 
                                                           placeholder='Enter name of corpus 1...', 
                                                           desc='')
        
        # widgets to select second source
        enter_source_2, source_2 = self.select_text_widget(value='Corpus 2:', 
                                                           placeholder='Enter name of corpus 2...', 
                                                           desc='')
        
        # widgets to select data transformation
        enter_trans, select_trans = self.select_options(instruction='Data transformation:',
                                                      options=['no transform',
                                                               'log transform',
                                                               'square root transform'],
                                                      value='no transform')
        
        # data transformation dictionary
        data_transform = {'no transform':[1,''],
                          'log transform':[0,'log'],
                          'square root transform':[0.5,'sqrt']}
        
        # widgets to select statistical test
        enter_stat, select_stat = self.select_options(instruction='Select statistical test:',
                                                      options=['Welch t-test',
                                                               'Fisher Permutation test'],
                                                      value='Fisher Permutation test')
        
        # widgets to select confidence interval (disabled for now)
        enter_conf, select_conf = self.select_options(instruction='Select confidence level:',
                                                      options=['90%',
                                                               '95%',
                                                               '99%'],
                                                      value='95%')
        
        # widget to display histogram
        hist_button, hist_out = self.click_button_widget(desc='Plot histogram',
                                                       margin='10px 35px 10px 0px',
                                                       width='152px')
        
        # function to define what happens when the hist button is clicked
        def on_hist_button_clicked(_):
            with stat_out:
                # clear output
                clear_output()
            with hist_out:
                # clear output
                clear_output()
                
                # capture selections
                w = word.value
                s1 = source_1.value
                s2 = source_2.value
                trans = select_trans.value
                
                # generate histogram as per selections
                try:
                    # perform word count
                    self.word_count(w)
                    
                    # define x and y for histogram
                    x, y = self.x_and_y(w, s1, s2, data_transform[trans][0])
                    
                    # plot the histogram
                    self.plot_histogram(x, y, w, s1, s2, data_transform[trans][1])
                
                # exception if the selected word does not exist in both corpora
                except:
                    print("The word '{}' does not exist in the selected corpora.".format(w))
        
        # link the hist button with the function
        hist_button.on_click(on_hist_button_clicked)
        
        # widget to display statistical test result
        stat_button, stat_out = self.click_button_widget(desc='Perform statistical analysis',
                                                       margin='10px 35px 10px 0px',
                                                       width='200px')
        
        # function to define what happens when the stat button is clicked
        def on_stat_button_clicked(_):
            with stat_out:
                # clear output
                clear_output()
                
                # capture selections
                w = word.value
                s1 = source_1.value
                s2 = source_2.value
                trans = select_trans.value
                which_stat = select_stat.value
                conf_level = select_conf.value
                
                # generate statistic as per selections
                try:
                    # perform word count
                    self.word_count(w)
                    
                    # define x and y for statistical analysis
                    x, y = self.x_and_y(w, s1, s2, data_transform[trans][0])
                    
                    # perform statisitical test
                    if which_stat=='Welch t-test':
                        stat, pvalue = self.welch_t_test(x, y)
                    else:
                        stat, pvalue = self.fisher_permutation_test(x, y)
                    
                    # print the output and analysis
                    print('\033[1m{}\033[0m'.format(which_stat))
                    print('Statistic score: {:.2f}'.format(stat))
                    print('p-value: {:.2f}'.format(pvalue))
                    print()
                    if stat>0: 
                        print("The mean frequency of the word '{}' is higher in '{}' than in '{}',".format(w, s1, s2))
                    else: 
                        print("The mean frequency of the word '{}' is lower in '{}' than in '{}',".format(w, s1, s2))
                    if pvalue<(1-int(conf_level.strip('%'))/100): 
                        print('and we consider the difference to be statistically significant.')
                        print("\nIn summary, we reject the null hypothesis that use of the word '{}' in '{}' is equal to that in '{}'.".format(w, s1, s2))
                    else: 
                        print('but the difference is not statistically significant.')
                        print("\nIn summary, we accept the null hypothesis that use of the word '{}' in '{}' is equal to that in '{}'.".format(w, s1, s2))
                
                # exception if the selected word does not exist in both corpora
                except:
                    print("The word '{}' does not exist in the selected corpora.".format(w))
        
        # link the stat button with the function
        stat_button.on_click(on_stat_button_clicked)
        
        # displaying inputs, buttons and their outputs
        hbox1 = widgets.HBox([enter_word, word],
                             layout = widgets.Layout(width='430px'))
        
        hbox2 = widgets.HBox([enter_source_1, source_1], 
                             layout = widgets.Layout(width='430px'))
        
        hbox3 = widgets.HBox([enter_source_2, source_2], 
                             layout = widgets.Layout(width='430px'))
        
        vbox1 = widgets.VBox([word_instruction,
                              hbox1,
                              corpora_instruction,
                              hbox2, 
                              hbox3], 
                             layout = widgets.Layout(width='430px'))
        
        vbox2 = widgets.VBox([enter_trans, 
                              select_trans,
                              hist_button], 
                             layout = widgets.Layout(width='230px'))
        
        vbox3 = widgets.VBox([enter_stat, select_stat,
                              stat_button], 
                             layout = widgets.Layout(width='260px'))
        
        hbox5 = widgets.HBox([vbox1, vbox2, vbox3], 
                            layout = widgets.Layout(height='185px'))
        
        vbox = widgets.VBox([hbox5, stat_out, hist_out])
        
        return vbox
        
        
    def enter_word_widget(self):
        '''
        TBC
        '''
        # widget to display instruction
        enter_word = widgets.HTML(
            value='Word:',
            placeholder='',
            description='',
            )
        
        # widget to enter the word
        word = widgets.Text(
            value='',
            placeholder='Enter a word...',
            description='',
            disabled=False,
            layout = widgets.Layout(width='210px')
        )
        
        return enter_word, word
    
    
    def select_text_widget(self, value: str, placeholder: str, desc: str):
        '''
        Create widgets for selecting text_name to analyse

        Args:
            entity: option to include 'all texts' for analysing top entities
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value=value,
            placeholder='',
            description='',
            )
        
        # get the list of corpora
        source_options = list(set(self.text_df.source.to_list()))
        
        # widget to display text_options
        text = widgets.Combobox(
            placeholder=placeholder,
            options=source_options,
            description=desc,
            ensure_option=True,
            disabled=False,
            layout = widgets.Layout(width='210px', height='20px')
        )
        
        return enter_text, text
    
    
    def select_options(self, 
                       instruction: str,
                       options: list,
                       value: str,
                       width: str = '180px'):
        '''
        Create widgets for selecting the number of entities to display
        
        Args:
            instruction: text instruction for user
            options: list of options for user
            value: initial value of the widget
            width: the width of the widget box
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widget to select entity options
        select_option = widgets.Dropdown(
            options=options,
            value=value,
            description='',
            disabled=False,
            layout = widgets.Layout(width=width)
            )
        
        return enter_text, select_option
    
    
    def select_multiple_options(self, 
                                instruction: str,
                                options: list,
                                value: list,
                                width: str):
        '''
        Create widgets for selecting muyltiple options
        
        Args:
            instruction: text instruction for user
            options: list of options for user
            value: initial value of the widget
            width: the width of the widget box
        '''
        # widget to display instruction
        enter_m_text = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widget to select entity options
        select_m_option = widgets.SelectMultiple(
            options=options,
            value=value,
            description='',
            disabled=False,
            layout = widgets.Layout(width=width)
            )
        
        return enter_m_text, select_m_option
        
        
    def click_button_widget(
            self, 
            desc: str, 
            margin: str='10px 0px 0px 10px',
            width='320px'
            ):
        '''
        Create a widget to show the button to click
        
        Args:
            desc: description to display on the button widget
            margin: top, right, bottom and left margins for the button widget
            width: the weidth of the button widget
        '''
        # widget to show the button to click
        button = widgets.Button(description=desc, 
                                layout=Layout(margin=margin, width=width),
                                style=dict(font_style='italic',
                                           font_weight='bold'))
        
        # the output after clicking the button
        out = widgets.Output()
        
        return button, out
    
    
    def select_n_widget(self, 
                        instruction: str, 
                        value: int,
                        min_value: int,
                        max_value: int):
        '''
        Create widgets for selecting the number of entities to display
        
        Args:
            instruction: text instruction for user
            value: initial value of the widget
            min_value: the minimum value of the widget
            max_value: the maximum value of the widget
        '''
        # widget to display instruction
        enter_n = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widgets for selecting n
        n_option = widgets.BoundedIntText(
            value=value,
            min=min_value,
            max=max_value,
            step=10,
            description='',
            disabled=False,
            layout = widgets.Layout(width='180px')
        )
        
        return enter_n, n_option
    
    
    def text_styling(self):
        '''
        Function for pandas dataframe text style
        '''
        # CSS styling 
        return """
        <style scoped>
            .dataframe-div {
              max-height: 250px;
              overflow: auto;
              position: relative;
            }
        
            .dataframe thead th {
              position: -webkit-sticky; /* for Safari */
              position: sticky;
              top: 0;
              background: #2ca25f;
              color: white;
            }
        
            .dataframe thead th:first-child {
              left: 0;
              z-index: 1;
            }
        
            .dataframe tbody tr th:only-of-type {
                    vertical-align: middle;
                }
        
            .dataframe tbody tr th {
              position: -webkit-sticky; /* for Safari */
              position: sticky;
              left: 0;
              background: #99d8c9;
              color: white;
              vertical-align: top;
            }
        </style>
        """