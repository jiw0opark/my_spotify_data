#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytz

df = pd.read_csv('/Users/jiwoopark/Downloads/spotify_listening.csv')

df.head()


print("Number of Values:", df.shape[0], "\nNumber of Features:", df.shape[1])

print("\nNumber of NA for each feature:\n")
print(df.isna().sum())

#get total minutes played by milliseconds to minutes by converting values
df['total_minutes_played'] = df['ms_played'] * 0.00001666666

#rename the column ts to timestamp
df.rename(columns={'ts':'timestamp'}, inplace=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
#df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

# timezone_map = {
#     'KR' : 'Asia/Seoul',
#     'US' : 'America/New_York'
# }

# def convert_timezone(row):
#     country = row['conn_country']
#     timezone = timezone_map.get(country)
#     return row['timestamp'].tz_convert(timezone)
    

# df['timestamp_local'] = df.apply(convert_timezone, axis=1)
# df['timestamp_local'] = pd.to_datetime(df['timestamp_local'], utc=True)

# print(df['timestamp_local'].head())

df['local_hour'] = df['timestamp'].dt.hour
df['local_date'] = df['timestamp'].dt.date


#replace the spotify track id to get rid of the prefix 'spotify:track:' to contain only the uri id
df['spotify_track_uri'] = df['spotify_track_uri'].str.replace('spotify:track:', '', regex=False)
df.rename(columns={"spotify_track_uri": "track_id"}, inplace = True)

#get rid of the nan columns
clean_df = df.drop(columns=['episode_name', 'episode_show_name', 
                            'spotify_episode_uri', 'skipped', 
                            'user_agent_decrypted', 'username'])

print("\nPreview of Data:\n", clean_df.head())

#number of na after cleaning data
print("\nNumber of NA for each feature after cleaning data:\n")
print(clean_df.isna().sum())

print("Number of Values:", clean_df.shape[0], "\nNumber of Features:", clean_df.shape[1])

clean_df.to_csv('cleaned_spotify.csv', index=False)


# In[121]:


plt.hist(df['total_minutes_played'],bins=5, color='steelblue', edgecolor='darkblue')
plt.title('Distribution of Listening Time (Minutes)')
plt.xlabel('Minutes Played')
plt.ylabel('Frequency')
plt.show()


# In[111]:


top10_songs = clean_df['master_metadata_track_name'].value_counts().head(10)
top10_songs.plot(kind = 'bar', color = 'palevioletred')
plt.title('Top 10 Most Played Songs')
plt.xlabel('Song Name')
plt.ylabel('Play Count')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[112]:


top7_artists = clean_df['master_metadata_album_artist_name'].value_counts().head(7)
top7_artists.plot(kind = 'bar', color = 'rosybrown')
plt.title('Top 7 Most Played Artist')
plt.xlabel('Artist Name')
plt.ylabel('Play Count')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[113]:


top5_albums = clean_df['master_metadata_album_album_name'].value_counts().head(5)
top5_albums.plot(kind = 'bar', color = 'darkseagreen')
plt.title('Top 5 Most Played Albums')
plt.xlabel('Album Name')
plt.ylabel('Play Count')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[122]:


# print(clean_df['timestamp'].head())

# clean_df['hour'] = clean_df['timestamp'].dt.hour

# loc_tz = pytz.timezone('US/Eastern')
# clean_df['timestamp'] = clean_df['timestamp'].dt.tz_convert(loc_tz)
# clean_df['hour'] = clean_df['timestamp'].dt.hour

print(clean_df['local_hour'].value_counts())

count_hourly = clean_df.groupby('local_hour').size()


plt.figure(figsize=(10,6))
count_hourly.plot(kind='line', marker='x', color='purple', linewidth=2)
plt.title('Music Listening Pattern by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Tracks Played')
plt.xticks(range(0, 24))  # Ensure all 24 hours are shown
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:




