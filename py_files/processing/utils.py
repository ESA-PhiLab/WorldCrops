


def clean_bavarian_labels(dataframe):
    df = dataframe.copy()

    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.Date = df.Date.dt.strftime('%m-%d')

    if df.NC.dtypes == 'int':

        df.loc[(df.NC == 600)|(df.NC == 601)|(df.NC == 602), 'NC'] = 1  #Kartoffeln jetzt alle Code 600
        df.loc[(df.NC == 131)|(df.NC == 476), 'NC'] = 2  #Wintergerste jetzt alle Code 131
        df.loc[(df.NC == 400)|(df.NC == 411)|(df.NC == 171)|(df.NC == 410)|(df.NC == 177), 'NC'] = 3  #Mais jetzt alle Code 400
        df.loc[(df.NC == 311)|(df.NC == 489), 'NC'] = 4  #Winterraps jetzt alle Code 311
        df.loc[(df.NC == 115), 'NC'] = 5  #WW
        df.loc[(df.NC == 603), 'NC'] = 6  #ZR

        df.loc[~((df.NC == 1)|(df.NC == 2)|(df.NC == 3)|(df.NC == 4)|(df.NC == 5)|(df.NC == 6)), 'NC'] = 0   #rejection class other

    else:
        print('Label type not int')
    
    return df