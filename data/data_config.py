def get_word_dict():
    ch_list = ['f', 'n', 'b', 'u', 'h', 'r', 'p', 'm', 't', 'e', 'a', ' ', 'o', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    word_dict = {ch_list[i]: i for i in range(len(ch_list))}

    return word_dict

if __name__ == '__main__':
    word_dict = get_word_dict()
    print(word_dict)