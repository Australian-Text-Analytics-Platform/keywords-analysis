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
import seaborn as sns

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
        

class KeywordAnalysis():
    '''
    Using word statistics to analyse words in a collection of corpus 
    and identify whether certain words are over or under-represented 
    in a particular corpus compared to their representation in other 
    corpus
    '''
    def __init__(self):
        '''
        Initiate the KeywordAnalysis
        '''
        # initiate other necessary variables
        self.large_file_size = 1000000
        self.text_df = pd.DataFrame()
        
        # create input and output folders if not already exist
        os.makedirs('input', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # initiate the variables for file uploading
        # widget for entering corpus name
        self.corpus_name = widgets.Text(
            value='',
            placeholder='Enter corpus name...',
            description='Corpus Name:',
            disabled=False,
            style= {'description_width': 'initial'}
        )
        
        # widget for file upload
        self.file_uploader = widgets.FileUpload(
            description='Upload your files (txt, csv, xlsx or zip)',
            accept='.txt, .xlsx, .csv, .zip', # accepted file extension
            multiple=True,  # True to accept multiple files
            error='File upload unsuccessful. Please try again!',
            layout = widgets.Layout(width='305px',
                                    margin='5px 0px 0px 0px')
            )
    
        self.upload_out = widgets.Output()
        
        # give notification when file is uploaded
        def _cb(change):
            with self.upload_out:
                # clear output and give notification that file is being uploaded
                clear_output()
                
                # check file size
                self.check_file_size(self.file_uploader)
                
                # reading uploaded files
                self.process_upload()
                
                # clear saved value in cache and reset counter
                self.file_uploader._counter=0
                self.file_uploader.value.clear()
                
                # give notification when uploading is finished
                print('Finished uploading files.')
                file_name = self.corpus_name.value
                if file_name!='':
                    print('{} text documents are loaded in corpus {}.'.format(self.text_df.shape[0], 
                                                                              file_name))
                else:
                    print('{} text documents are loaded.'.format(self.text_df.shape[0]))
                
                self.corpus_name.value = ''
                self.corpus_name.placeholder='Enter corpus name...'
                print('\nYou can now upload your next corpus, or continue to the next step')
            
        # observe when file is uploaded and display output
        self.file_uploader.observe(_cb, names='data')
        self.upload_box = widgets.VBox([self.corpus_name, 
                                        self.file_uploader, 
                                        self.upload_out])
        
        # CSS styling 
        self.style = """
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
        
        
    def check_file_size(self, file):
        '''
        Function to check the uploaded file size
        
        Args:
            file: the uploaded file containing the text data
        '''
        # check total uploaded file size
        total_file_size = sum([i['metadata']['size'] for i in self.file_uploader.value.values()])
        print('The total size of the upload is {:.2f} MB.'.format(total_file_size/1000000))
        
        # display warning for individual large files (>1MB)
        large_text = [text['metadata']['name'] for text in self.file_uploader.value.values() \
                      if text['metadata']['size']>self.large_file_size and \
                          text['metadata']['name'].endswith('.txt')]
        if len(large_text)>0:
            print('The following file(s) are larger than 1MB:', large_text)
        
        
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
            print('Extracting {}...'.format(zip_file['metadata']['name']))
            zip.extractall('./input/')
        
        # clear up temp
        temp = None
    
    
    def load_txt(self, file) -> list:
        '''
        Load individual txt file content and return a dictionary object, 
        wrapped in a list so it can be merged with list of pervious file contents.
        
        Args:
            file: the file containing the text data
        '''
        try:
            # read the unzip text file
            with open(file) as f:
                temp = {'text_name': file.name[:-4],
                        'text': f.read()
                }
            os.remove(file)
        except:
            file = self.file_uploader.value[file]
            # read and decode uploaded text
            temp = {'text_name': file['metadata']['name'][:-4],
                    'text': codecs.decode(file['content'], encoding='utf-8', errors='replace')
            }
            
            # check for unknown characters and display warning if any
            unknown_count = temp['text'].count('�')
            if unknown_count>0:
                print('We identified {} unknown character(s) in the following text: {}'.format(unknown_count, file['metadata']['name'][:-4]))
        
        return [temp]


    def load_table(self, file) -> list:
        '''
        Load csv or xlsx file
        
        Args:
            file: the file containing the excel or csv data
        '''
        if type(file)==str:
            file = self.file_uploader.value[file]['content']

        # read the file based on the file format
        try:
            temp_df = pd.read_csv(file)
        except:
            temp_df = pd.read_excel(file)
        
        # remove file from directory
        if type(file)!=bytes:
            os.remove(file)
            
        # check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('text' not in temp_df.columns) \
            or ('text_name' not in temp_df.columns) \
                or ('source' not in temp_df.columns):
            print('File {} does not contain the required header "text", "text_name" and "source"'.format(file['metadata']['name']))
            return []
        
        # return a list of dict objects
        temp = temp_df[['text_name', 'text', 'source']].to_dict(orient='index').values()
        
        return temp
    
    
    def hash_gen(self, temp_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create column text_id by md5 hash of the text in text_df
        
        Args:
            temp_df: the temporary pandas dataframe containing the text data
        '''
        #temp_df['text_id'] = temp_df['text'].apply(lambda t: hashlib.md5(t.encode('utf-8')).hexdigest())
        temp_df['text_id'] = temp_df['text'].apply(
            lambda t: hashlib.shake_256(t.encode('utf-8')).hexdigest(5))
        
        return temp_df
    
    
    def process_upload(self, deduplication: bool = True):
        '''
        Pre-process uploaded .txt files into pandas dataframe

        Args:
            deduplication: option to deduplicate text_df by text_id
        '''
        # create placeholders to store all texts and zipped file names
        all_data = []; zip_files = []
        
        # read and store the uploaded files
        files = list(self.file_uploader.value.keys())
        
        # extract zip files (if any)
        for file in files:
            if file.lower().endswith('zip'):
                self.extract_zip(self.file_uploader.value[file])
                zip_files.append(file)
        
        # remove zip files from the list
        files = list(set(files)-set(zip_files))
        
        # add extracted files to files
        for file_type in ['*.txt', '*.xlsx', '*.csv']:
            files += [file for file in Path('./input').rglob(file_type) if 'MACOSX' not in str(file)]
        
        print('Reading uploaded files...')
        print('This may take a while...')
        # process and upload files
        for file in tqdm(files):
            # process text files
            if str(file).lower().endswith('txt'):
                text_dic = self.load_txt(file)
                    
            # process xlsx or csv files
            else:
                text_dic = self.load_table(file)
            all_data.extend(text_dic)
        
        # remove files and directory once finished
        os.system('rm -r ./input')
        
        # convert them into a pandas dataframe format and add unique id
        temp_df = pd.DataFrame.from_dict(all_data)
        if 'source' not in temp_df.columns:
            temp_df['source'] = len(temp_df) * [self.corpus_name.value]
        temp_df = self.hash_gen(temp_df)
        
        # clear up all_data
        all_data = []; zip_files = []
        
        self.text_df = pd.concat([self.text_df,temp_df])

        # deduplicate the text_df by text_id
        if deduplication:
            self.text_df.drop_duplicates(subset='text_id', keep='first', inplace=True)
        
    
    def calculate_word_statistics(self):
        '''
        Function to calculate word statistics in a collection of corpus
        The statistics include normalised word count, log-likelihood, 
        Bayes factor BIC, effect size for log-likelihood (ELL), 
        relative risk, log ratio and odds ratio. For more information, 
        please visit this website: https://ucrel.lancs.ac.uk/llwizard.html
        '''
        # collate all texts based on source and 
        # use CountVectorizer to count the number of words in each source
        self.wordcount_df = collapse_corpus_by_source(df=self.text_df)
        self.wordcount_df = count_words(df=self.wordcount_df)
        
        # get total word counts based on source and overall in the corpus
        self.wordcount_df, total_by_source, total_words_in_corpus = get_totals(df=self.wordcount_df)
        
        # pairwise comparison between a particular corpus vs the rest of the corpus
        self.pairwise_compare = two_corpus_compare(self.wordcount_df, 
                                                   total_by_source, 
                                                   total_words_in_corpus)
        
        self.all_words = self.pairwise_compare.word.to_list()
        
        # multi-corpora analysis
        self.multicorp_comparison = n_corpus_compare(self.wordcount_df, 
                                                     total_by_source, 
                                                     total_words_in_corpus)
        
        
    def visualize_stats(self, 
                        df: pd.DataFrame, 
                        yticks,
                        index: int, 
                        inc_chart: list, 
                        title: str, 
                        last_chart: bool, 
                        figsize: tuple, 
                        bbox_to_anchor: tuple,
                        multi: bool = False):
        '''
        Function to visualize the calculated statistics onto a line chart

        Args:
            df: the pandas dataframe containing the selected data 
            yticks: the calculated yticks for the line chart 
            index: the index of the first word 
            inc_chart: the list of statistics to be included in the chart 
            title: the title of the chart, 
            last_chart: whether it is the last chart or not, 
            figsize: the size of the chart, 
            bbox_to_anchor: the location of the legend box,
            multi: whether the chart is for multi-corpora analysis or not
        '''
        # define the parameters for the chart
        plt.figure(figsize=figsize)
        sns.set_theme(style="whitegrid")
        plt.margins(x=0.025, tight=True)
        plt.title(title)
        
        # define the data
        data = df.iloc[index:index+40,inc_chart]
        
        # create the line chart
        fig = sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        
        # if last chart include words in the x-ticks + legend on the right
        if last_chart:
            plt.xticks(rotation=90, 
                       fontsize='small')
            fig.legend(loc='right', 
                       bbox_to_anchor=bbox_to_anchor, 
                       ncol=1, 
                       fontsize='small')
        else:
            plt.xticks(ticks=range(0,40),
                       labels=['']*40)
            fig.legend('')
            
        # define x-label and y-ticks
        plt.xlabel('')
        #plt.yticks(yticks)
        
        return fig
    
    
    def set_yticks(self, 
                   df: pd.DataFrame, 
                   inc_chart: list, 
                   inc_corpus: list = None, 
                   inc_col: list = None, 
                   multi: bool = True):
        '''
        Function to calculate the minimum and maximum values for the y-ticks
        and generate a numpy array for the y-ticks

        Args:
            df: the pandas dataframe containing the selected data 
            inc_chart: the list of statistics to be included in the chart 
            inc_corpus: the list of corpus to be included in the chart 
            inc_col: the list of columns to be included in the chart
            multi: whether the chart is for multi-corpora analysis or not
        '''
        # create placeholders for minimum and maximum values
        max_values = []; min_values = []
        
        # define the minimum and maximum values baed on the data
        if multi:
            for chart in inc_chart:
                max_values.append(max(df.iloc[:,chart].to_list()))
                min_values.append(min(df.iloc[:,chart].to_list()))
        else:
            for corpus in inc_corpus:
                for col in inc_col:
                    max_values.append(max(df[col+corpus].to_list()))
                    min_values.append(min(df[col+corpus].to_list()))
        max_value = max(round(max(max_values),0),0.05)
        min_value = min(round(min(min_values),0),-0.05)
        
        # generate y-ticks based on calculated minimum and maximum values
        if not(max_value<1 and min_value>-1):
            if max_value%5!=0:
                max_value=(max_value//5+1)*5
            if min_value%5!=0:
                if min_value>0:
                    min_value=(min_value//5+1)*5
                else:
                    min_value=((min_value//5))*5
            yticks=np.linspace(int(min_value), int(max_value), int((max_value-min_value)/5+1))
        else:
            yticks=np.linspace(min_value, max_value, 5)
        
        return yticks
    
    
    def create_graphs(self, 
                      viz_df: pd.DataFrame, 
                      index: int, 
                      inc_corpus: list, 
                      inc_charts: list, 
                      options: dict(), 
                      multi: bool):
        '''
        Function to generate line charts based on selected parameters

        Args:
            viz_df: the pandas dataframe containing the selected data 
            index: the index of the first word 
            inc_corpus: the list of corpus to be included in the chart 
            inc_charts: the list of statistics to be included in the chart 
            options: the dictionary containing the statistic options to display 
            multi: whether the chart is for multi-corpora analysis or not 
        '''
        inc_chart = [options[chart][0] for chart in inc_charts]
        
        if multi:
            #yticks = self.set_yticks(self.multicorp_comparison, inc_chart)
            yticks=None
            last_chart = True
            which_corpus = 'multi-corpus'
            figsize=(8, 4)
            bbox_to_anchor=(1.3, 0.5)
            fig = self.visualize_stats(viz_df, 
                                       yticks,
                                       index, 
                                       inc_chart, 
                                       which_corpus, 
                                       last_chart,
                                       figsize, 
                                       bbox_to_anchor,
                                       multi)
            plt.show()
        else:
            inc_col = [options[chart][1] for chart in inc_charts]
            yticks=None
            #yticks = self.set_yticks(self.pairwise_compare, 
            #                         inc_chart, 
            #                         inc_corpus, 
            #                         inc_col, 
            #                         multi=False)
            
            # display bar chart for every selected entity type
            for n, which_corpus in enumerate(inc_corpus):
                selected_corpus = [column for column in viz_df.columns.to_list() \
                                   if which_corpus in column]
                df = viz_df.loc[:,selected_corpus]
                
                last_chart = False
                if n==(len(inc_corpus)-1):
                    last_chart = True
                figsize=(8, 3)
                bbox_to_anchor=(1.7, 0.5)
                fig = self.visualize_stats(df, 
                                           yticks,
                                           index, 
                                           inc_chart, 
                                           which_corpus, 
                                           last_chart, 
                                           figsize, 
                                           bbox_to_anchor,
                                           multi)
                plt.show()
        
    def analyse_stats(
            self, 
            multi: bool = False
            ):
        '''
        Function to generate widgets for analysing calculated statistics

        Args:
            multi: whether the chart is for multi-corpora analysis or not 
        '''
        # define corpus list
        corpus_options = list(set(self.text_df.source))
        
        # widget to select corpus to include in the analysis
        enter_corpus, select_corpus = self.select_multiple_options('<b>Select corpus:</b>',
                                                               corpus_options,
                                                               corpus_options,
                                                               '150px')
        
        # whether to do pairwise or multi-corpora analysis
        if multi:
            viz_df = self.multicorp_comparison.copy()
            options = {'log-likelihood':[-3],
                       'bayes factor BIC':[-2],
                       'ELL':[-1]}
        else:
            viz_df = self.pairwise_compare.copy()
            options = {'normalised word count (corpus)':[3,'normalised_wc_'],
                       'normalised word count (rest of corpus)':[4,'normalised_restofcorpus_wc_'],
                       'log-likelihood':[6,'log_likelihood_'],
                       'bayes factor BIC':[8,'bayes_factor_bic_'],
                       'ELL':[9,'ell_'],
                       'relative risk':[10,'relative_risk_'],
                       'log ratio':[11,'log_ratio_'],
                       'odds ratio':[12,'odds_ratio_']}
        # set the words as the index of the dataframe
        viz_df.set_index('word', inplace=True)
            
        # widget to select statistics to be included in the line chart
        enter_chart, select_chart = self.select_multiple_options('<b>Select statistic(s) to display):</b>',
                                                               list(options.keys()),
                                                               list(options.keys()),
                                                               '250px')
        
        # widget to display analysis
        display_button, display_out = self.click_button_widget(desc='Display chart',
                                                       margin='0px 35px 0px 0px',
                                                       width='152px')
        
        # selection slider to select words in the corpus
        display_index = widgets.SelectionSlider(
            options=self.all_words[:-40],
            value=self.all_words[0],
            description='First word:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            layout = widgets.Layout(width='280px')
        )
        
        # update charts when the slider is moved
        def _cb(change):
            with display_out:
                # clear output and get word index
                clear_output(wait=False)
                index= self.all_words.index(display_index.value)
                
                # display updated charts
                self.create_graphs(viz_df, 
                                   index, 
                                   select_corpus.value, 
                                   select_chart.value, 
                                   options, 
                                   multi)
        
        # observe when selection slider is moved
        display_index.observe(_cb, names='value')
        
        # function to define what happens when the display button is clicked
        def on_display_button_clicked(_):
            with display_out:
                # clear output and get word index
                clear_output()
                index=0
                
                # display updated charts
                self.create_graphs(viz_df, 
                                   index, 
                                   select_corpus.value, 
                                   select_chart.value, 
                                   options, 
                                   multi)
        
        # link the display button with the function
        display_button.on_click(on_display_button_clicked)
        
        # displaying inputs, buttons and their outputs
        vbox1 = widgets.VBox([enter_corpus,
                              select_corpus], 
                             layout = widgets.Layout(width='200px', height='150px'))
        vbox2 = widgets.VBox([enter_chart, 
                              select_chart], 
                             layout = widgets.Layout(width='350px', height='150px'))
        
        # exclude corpus selection for multi-corpora analysis
        if multi:
            hbox1 = widgets.HBox([vbox2])
        else:
            hbox1 = widgets.HBox([vbox1, vbox2])
            
        hbox2 = widgets.HBox([display_button, display_index])
        
        vbox = widgets.VBox([hbox1, hbox2, display_out])
        
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
        # display the first n rows in html format
        df_html = df.head(display_n).to_html(escape=False)
        
        # apply css styling
        df_html = self.style+'<div class="dataframe-div">'+df_html+"\n</div>"
        
        # display analysis
        display(HTML(df_html))
        
        # save analysis into an Excel spreadsheet
        values = [df.columns] + list(df.values)
        wb = Workbook()
        wb.new_sheet(sheet_name, data=values)
        wb.save(output_dir + file_name)
        
        # download the file onto your computer
        print('Click below to download:')
        display(DownloadFileLink(output_dir + file_name, file_name))
        
        
    def select_options(self, 
                       instruction: str,
                       options: list,
                       value: str):
        '''
        Create widgets for selecting the number of entities to display
        
        Args:
            instruction: text instruction for user
            options: list of options for user
            value: initial value of the widget
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
            layout = widgets.Layout(width='150px')
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