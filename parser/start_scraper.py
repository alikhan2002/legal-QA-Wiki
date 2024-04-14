import os


class Starter:

    @classmethod
    def start_spider(cls):
        
        if not os.path.exists('dataset'):
            os.mkdir('dataset')

        os.system(f"scrapy crawl AdiletZan -o dataset/all_dataset.json")



if __name__ == '__main__':
    Starter.start_spider()