import numpy as np
import pandas as pd
import os
import re

def unique_jic_list():
    dfjic = pd.read_csv('NRDataKLM/JIC anonimised TUD 1.csv')
    alljic = dfjic['jic_code']
    print(len(alljic))
    uniquejic = set(alljic)
    return uniquejic

def process_jiccode(jiccode, dfjic, dfnr, dfwp):

    #Filter out needed jic based on jic_code in JIC dataframe
    escaped_jiccode = re.escape(jiccode)  # Escapes any special regex characters
    dfjic1 = dfjic[dfjic['jic_code'].str.contains(escaped_jiccode)]
    #Initialise nrlist
    nrlist = np.array([[0, 0, '', 0, 0]])

    # Sort for more than 1000

    if dfjic1.shape[0] == 10 and not os.path.exists('RT_datasets/' +jiccode+'.csv'):
        for i, row in enumerate(dfjic1.itertuples()):
            #Extract some elements
            print(f'->{i}/{dfjic1.shape[0]}')
            taskbarcode = row[4]
            wpbarcode = row[3]

            dfwp1 = dfwp[dfwp['wp_wp_barcode'] == wpbarcode]
            type_val = dfwp1.iloc[0, 8]
            flycycle = row[12]
            flyhour = row[13]
            #Following processing happens in NR dataframe not JIC dataframe
            #First , filter out nr task based on the source(comes from which specific task)
            dfnr1 = dfnr[dfnr['NR_found_in_barcode'].str.contains(taskbarcode, na=False)]
            #Then , filter out nr task that is actually carried out in its workpackage
            dfnr2 = dfnr1[dfnr1['nr_start_datetime'] >= dfnr1['nr_wp_actual_start']]
            #And , sort the nr task according to time in accending order
            dfnr3 = dfnr2.loc[dfnr2.groupby('nr_barcode')['nr_start_datetime'].idxmin()]
            #Only keep the first record of each distinctive nr task
            a = list(dfnr3['nr_barcode'].unique())
            #NR task summerise
            task_number = len(a)
            labour = dfnr3['nr_actual_labour_hours'].sum()
            #Put into array
            nrlist = np.row_stack((nrlist, [task_number, labour, type_val, flycycle, flyhour]))
        # Put into dataframe
        dfjic1['nrtask'] = nrlist[1:, 0]
        dfjic1['nrlabour'] = nrlist[1:, 1]
        dfjic1['type'] = nrlist[1:, 2]
        dfjic1['flycycle'] = nrlist[1:, 3]
        dfjic1['flyhour'] = nrlist[1:, 4]
        if '/' in jiccode:
            jiccode = jiccode.replace('/', '_')  # Replace '/' with '_'
            dfjic1.to_csv('RT_datasets/' + jiccode + '.csv')
        else:
            dfjic1.to_csv('RT_datasets/' +jiccode+'.csv')

    

#Read data
jic = pd.read_csv('NRDataKLM/JIC anonimised TUD 1.csv')
nr = pd.read_csv('NRDataKLM/NR anonimised TUD.csv')
wp = pd.read_csv('NRDataKLM/WP anonimised TUD.csv', delimiter=';')
#Datetime format correction
jic['jic_start_datetime'] = pd.to_datetime(jic['jic_start_datetime'],format="%d/%m/%Y %H:%M")
nr['nr_start_datetime'] = pd.to_datetime(nr['nr_start_datetime'],format="%d/%m/%Y %H:%M")
nr['nr_end_datetime'] = pd.to_datetime(nr['nr_end_datetime'],format="%d/%m/%Y %H:%M")
nr['nr_wp_actual_start'] = pd.to_datetime(nr['nr_wp_actual_start'],format="%d/%m/%Y %H:%M")
nr['nr_wp_actual_end'] = pd.to_datetime(nr['nr_wp_actual_end'],format="%d/%m/%Y %H:%M")


ujiclst = unique_jic_list()
for i, code in enumerate(ujiclst):
    print(f'({i}/{len(ujiclst)})')
    process_jiccode(code, jic, nr, wp)
