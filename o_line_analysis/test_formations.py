import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set()

PATH = "oline_data"

df_flattened = pd.read_csv(f"{PATH}/df_flattened_w_preds.csv")
scouting = pd.read_csv(f"{PATH}/data/pffScoutingData.csv")
plays = pd.read_csv(f"{PATH}/data/plays.csv")

def get_play(df_flattened, game_id, play_id):
    play = df_flattened[(df_flattened['gameId'] == game_id) & (df_flattened['playId'] == play_id)]

    def_df = pd.DataFrame()
    def_df['x'] = play[[f"x_{num}" for num in range(1, 12)]].squeeze().to_list()
    def_df['y'] = play[[f"y_{num}" for num in range(1, 12)]].squeeze().to_list()
    def_df['nfl_id'] = play[[f"nflId_{num}" for num in range(1, 12)]].squeeze().to_list()
    predictions = play[[f"prediction_{num}" for num in range(1, 12)]].squeeze() > 0.5
    def_df['prediction'] = predictions.to_list()
    def_df['isRusher'] = play[[f"isRusher_{num}" for num in range(1, 12)]].squeeze().astype(bool).to_list()
    

    off_df = pd.DataFrame()
    off_df['x'] = play[[f"x_{num}" for num in range(13, 24)]].squeeze().to_list()
    off_df['y'] = play[[f"y_{num}" for num in range(13, 24)]].squeeze().to_list()
    off_df['nfl_id'] = play[[f"nflId_{num}" for num in range(13, 24)]].squeeze().to_list()

    play_scouting = scouting[(scouting['gameId'] == game_id)\
                              & (scouting['playId'] == play_id)]
    pos_mapper = play_scouting.set_index('nflId')['pff_positionLinedUp']

    def_df['pos'] = def_df['nfl_id'].map(pos_mapper)
    off_df['pos'] = off_df['nfl_id'].map(pos_mapper)

    return def_df, off_df

def plot_players(def_df, off_df, color):

    fig, ax = plt.subplots()
    sns.scatterplot(x='x', y='y', data=def_df, hue=color, s=400)
    sns.scatterplot(x='x', y='y', data=off_df, color=sns.color_palette("muted")[7], s=400)

    rect = patches.Rectangle((0, -7.5), 2.5, 15, facecolor='none', edgecolor='red')

    ax.add_patch(rect)

    # loop through each x,y pair
    for i, pos in enumerate(off_df['pos']):
        ax.annotate(off_df['pos'][i],  xy=(off_df['x'][i], off_df['y'][i]), color='black',
                    fontsize="small", weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')
        
    for i, pos in enumerate(def_df['pos']):
        ax.annotate(def_df['pos'][i],  xy=(def_df['x'][i], def_df['y'][i]), color='black',
                    fontsize="small", weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')
        
    plt.legend(title=color)

    plt.show()

def get_vert_table(df_flattened):
    vert_table = pd.DataFrame()

    vert_table['x_def'] = df_flattened[[f"x_{num}" for num in range(1, 12)]].to_numpy().flatten()
    vert_table['y_def'] = df_flattened[[f"y_{num}" for num in range(1, 12)]].to_numpy().flatten()
    vert_table['nfl_id_def'] = df_flattened[[f"nflId_{num}" for num in range(1, 12)]].to_numpy().flatten()
    vert_table['predictions'] = df_flattened[[f"prediction_{num}" for num in range(1, 12)]].to_numpy().flatten()
    vert_table['isRusher'] = df_flattened[[f"isRusher_{num}" for num in range(1, 12)]].to_numpy().flatten()
    vert_table['game_id'] = df_flattened['gameId'].repeat(11).reset_index(drop=True)
    vert_table['play_id'] = df_flattened['playId'].repeat(11).reset_index(drop=True)
    vert_table['part_of_front'] = (vert_table['x_def'] < 2.5) & (vert_table['y_def'].abs() < 7.5)

    return vert_table

def get_new_baseline(vert_table):
    baseline_acc = accuracy_score(vert_table['isRusher'], vert_table['part_of_front'].astype(int))

    print(baseline_acc)


def calc_disguise_scores(play):
    rushers = play[play['isRusher'] == 1]
    disguise_score = (1 - rushers['predictions']).mean() * 100

    return disguise_score

def get_front_by_play(vert_table):
    score_by_play = pd.DataFrame(vert_table.groupby(['game_id', 'play_id']).apply(calc_disguise_scores), columns=['disguise_score'])
    score_by_play['down_linemen'] = vert_table.groupby(['game_id', 'play_id']).agg(np.sum)['part_of_front']
    score_by_play['rushers'] = vert_table.groupby(['game_id', 'play_id']).agg(np.sum)['isRusher']

    return score_by_play
if __name__ == "__main__":
    game_id, play_id = df_flattened.iloc[3][['gameId', 'playId']]
    def_df, off_df = get_play(df_flattened, game_id, play_id)
    plot_players(def_df, off_df, 'prediction')
    plot_players(def_df, off_df, 'isRusher')
    vert_table = get_vert_table(df_flattened)
    get_new_baseline(vert_table)