from utils import *

if __name__ == '__main__':
    curr_path = os.path.dirname(os.path.realpath(__file__))
    path = curr_path+'/Corpus/e961024.htm'


    if not Path('Full_text_normalized.pkl').is_file():
        Full_text_normalized = getFullNormalize(path)
        save_data(Full_text_normalized, "Full_text_normalized.pkl")
    else:
        Full_text_normalized = retrieve_data("Full_text_normalized.pkl")
        print("Full_text_normalized file: OK, len:",len(Full_text_normalized))

    if not Path('articles.pkl').is_file():
        articles = getArticles(path)
        save_data(articles, "articles.pkl")
    else:
        articles = retrieve_data("articles.pkl")
        print("Articles file: OK, len:",len(articles))

    if not Path('nouns.pkl').is_file():
        nouns = getNounsList(Full_text_normalized)
        save_data(nouns, "nouns.pkl")
    else:
        nouns = retrieve_data("nouns.pkl")
        print("Nouns file: OK, len:",len(nouns))

    
    if not Path('idfdf.pkl').is_file():
        idfdf = getNounsListIDFTF(Full_text_normalized)
        save_data(idfdf, "idfdf.pkl")
    else:
        idfdf = retrieve_data("idfdf.pkl")
        print("idfdf file: OK, len:",len(idfdf))

    topics =["crisis","privatización", "contaminación", "política", "economía", "tecnología", "televisa"]

    topic_covarage = topic_covarage(topics, articles)




    print(np.array2string(topic_covarage, separator=', '))