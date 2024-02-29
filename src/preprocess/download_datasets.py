from datasets import load_dataset

if __name__ == '__main__':
    # dataset = load_dataset("wiki_bio") 
    # dataset.save_to_disk("./data/wikibio")

    dataset = load_dataset("wiki_asp", "artist") 
    dataset.save_to_disk("./data/wikiasp/artist")
    dataset = load_dataset("wiki_asp", "soccer_player") 
    dataset.save_to_disk("./data/wikiasp/soccer_player")