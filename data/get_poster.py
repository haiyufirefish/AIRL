import csv
import urllib.parse
from urllib.error import HTTPError
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import urllib.request



#This class is for pictures capturing
def get_url():
    row_names = ['movie_id', 'movie_title']
    with open('u.item', 'r', encoding = "ISO-8859-1") as f:
        reader = csv.DictReader(f, fieldnames=row_names, delimiter='|')
        for row in reader:
            movie_id = row['movie_id']
            movie_title = row['movie_title']
            domain = 'http://www.imdb.com'
            search_url = domain + '/find?q=' + urllib.parse.quote_plus(movie_title)
            with urllib.request.urlopen(search_url) as response:
                html = response.read()
                soup = BeautifulSoup(html, 'html.parser')
                # Get url of 1st search result
                try:
                    title = soup.find('table', class_='findList').tr.a['href']
                    movie_url = domain + title
                    with open('movie_urls.csv', 'a', newline='') as out_csv:
                        writer = csv.writer(out_csv, delimiter=',')
                        writer.writerow([movie_id, movie_url])
                # Ignore cases where search returns no results
                except AttributeError:
                    pass

def get_pos():
    row_names = ['movie_id', 'movie_url']
    with open('movie_urls.csv', 'r', newline='') as in_csv:
        reader = csv.DictReader(in_csv, fieldnames=row_names, delimiter=',')
        for row in tqdm(reader):
            movie_id = row['movie_id']
            movie_url = row['movie_url']
            domain = 'http://www.imdb.com'
            with urllib.request.urlopen(movie_url) as response:
                html = response.read()
                soup = BeautifulSoup(html, 'html.parser')
                # Get url of poster image
                try:
                    image_url = soup.find('div', class_='poster').a.img['src']
                    # TODO: Replace hardcoded extension with extension from string itself
                    extension = '.jpg'
                    extension = '.' + image_url.split('.')[-1]
                    image_url = ''.join(image_url.partition('_')[0]) + extension
                    filename = 'img/' + movie_id + extension
                    with urllib.request.urlopen(image_url) as response:
                        with open(filename, 'wb') as out_image:
                            out_image.write(response.read())
                        with open('movie_poster.csv', 'a', newline='') as out_csv:
                            writer = csv.writer(out_csv, delimiter=',')
                            writer.writerow([movie_id, image_url])
                    print(image_url)
                # Ignore cases where no poster image is present
                except AttributeError:
                    pass


def save_image():
    df = pd.read_csv('movie_poster_urls.csv', header = None, names=['movieId', 'url'])
    for ind in df.index:
        try:
            urllib.request.urlretrieve(
                "{}".format(df['url'][ind]),
                "./poster/{}.jpg".format(df['movieId'][ind]))
        except HTTPError:
            pass

def construct_link():
    df = pd.read_csv('movie_urls.csv', names = ['movieId', 'link'], header = None)
    imdbIds = []
    for ind in df.index:
        s = df['link'][ind][-9:-1]
        if s.isnumeric():
            imdbIds.append(s)
        else:
            imdbIds.append(df['link'][ind][-8:-1])
        # print(df['link'][ind][-8:-1])

    df['imdbId'] = imdbIds
    dff = df[['movieId','imdbId']]
    dff.to_csv('links.csv',index = False)
    print(dff)



if __name__ == '__main__':
    df = pd.read_csv('links.csv', names=['movieId', 'imdbId'], header = 0)

    print(type(df['movieId'][0]))
    count = 0
    numbers = np.arange(1,1683,1,dtype = np.int64).tolist()

    for ind in df.index:
        print(df['movieId'][ind])

        numbers.remove(df['movieId'][ind])

    print(numbers)
    movie_df = pd.read_csv("./ml-100k/u.item", sep="|", encoding='latin-1', header=None)
    movie_df.columns = ['movieId', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown',
                        'Action',
                        'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir',
                        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    for n in numbers:
        print(movie_df[movie_df['movieId'] == n]['title'])


