from datetime import datetime
import random


def clean_bavarian_labels(dataframe):
    df = dataframe.copy()

    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    if 'Date' in df.columns:
        df['Year'] = df.Date.dt.year
        df.Date = df.Date.dt.strftime('%m-%d')
        
    df.NC = df.NC.astype(str).astype(int)

    df.loc[(df.NC == 601)|(df.NC == 602), 'NC'] = 600  #Potato
    df.loc[(df.NC == 131)|(df.NC == 476), 'NC'] = 131  #Winter barley
    df.loc[(df.NC == 411)|(df.NC == 171)|(df.NC == 410)|(df.NC == 177), 'NC'] = 400  #Corn
    df.loc[(df.NC == 311)|(df.NC == 489), 'NC'] = 311  #Winter rapeseed
    #WW = 115
    #ZR = 603
    df.loc[~((df.NC == 600)|(df.NC == 131)|(df.NC == 400)|(df.NC == 311)|(df.NC == 115)|(df.NC == 603)), 'NC'] = 0  #rejection class Other

    #rewrite classes for easy handling
    df.loc[(df.NC == 600), 'NC'] = 1 
    df.loc[(df.NC == 131), 'NC'] = 2 
    df.loc[(df.NC == 400), 'NC'] = 3 
    df.loc[(df.NC == 311), 'NC'] = 4 
    df.loc[(df.NC == 115), 'NC'] = 5 
    df.loc[(df.NC == 603), 'NC'] = 6

    return df

def get_other_years(currentyear, yearlist):
    if currentyear not in yearlist:
        print('Year not in list')
        return currentyear
    else:
        yearlist = yearlist.copy()
        yearlist.remove(currentyear)
        output = random.sample(yearlist, len(yearlist))
        return output[0],output[1]

def augment_df(_df, years):
    ''' 
    dataframe: _df needs 'id', 'NC' and 'Year' column
    years: amount of years for augmentation
    '''

    if not ('NC' in _df.columns) & ('id' in _df.columns) & ('Year' in _df.columns):
        print('Verify the necessary columns NC, id, year')
        return

    train = _df.copy()
    groups = train.groupby('id')
    for id, group in groups:
        croptype = group['NC'].values[0]
        year = group['Year'].values[0]
        year1,year2 = get_other_years(year, years)

        try:
            #filter by potential fields ..same crop type/ different years
            augment1 = train[(train.NC == croptype)&(train['Year'] == year1)]
            augment2 = train[(train.NC == croptype)&(train['Year'] == year2)]

            #in case no data for other than current year is available
            if len(augment1) == 0:
                augment1 = train[(train.NC == croptype)&(train['Year'] == year)]
                augment2 = train[(train.NC == croptype)&(train['Year'] == year)]

            #get random id 
            _id = random.choice(list(set(augment1.id.values)))
            _id2 = random.choice(list(set(augment2.id.values)))

            x1 = augment1[augment1.id == int(_id)]
            x2 = augment2[augment2.id == int(_id2)]

            train.loc[train.id == id, 'id_x1'] = int(x1.head(1).id.values[0])
            train.loc[train.id == id, 'id_x2'] = int(x2.head(1).id.values[0])

        except:
            print('Error in Augmentation')
            pass

    return train

def get_embeddings_plot():
    """Creates a scatter plot with image overlays.
    """
    # initialize empty figure and add subplot
    fig = plt.figure()
    fig.suptitle('Scatter Plot of the Sentinel-2 Dataset')
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1., 1.]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    for idx in shown_images_idx:
        circle = plt.Circle((embeddings_2d[idx][0], embeddings_2d[idx][1]), 0.02, color='r')
        ax.add_artist(circle)

    # set aspect ratio
    ratio = 1. / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable='box')