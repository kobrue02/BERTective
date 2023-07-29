import json

with open("data.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

wip_author_dict = {}

try:
    # try to open existing file
    with open("author_dict.json", "r", encoding="UTF-8") as f:
        author_book_items = json.load(f)["books"]
        current_status = len(author_book_items)

except FileNotFoundError:
    # if running for first time
    author_book_items = []
    current_status = 0

if len(author_book_items) > 0:
    # if file was found and has data in it
    last_book = author_book_items[len(author_book_items)-1]["book_name"]
    last_author = author_book_items[len(author_book_items)-1]["author_name"]
else:
    last_book = ""
    last_author = ""


for i, item in enumerate(data["texts"]):

    # find stored items
    books = [item["book_name"] for item in author_book_items]
    authors = [item["author_name"] for item in author_book_items] 
    try:
        name = item["author_name"]
        book = item["book_name"]
        if name in authors:
            if book in books:
                # if book has been annotated already
                continue
            else:
                last_book = book
                gender = author_book_items[current_status-1]["author_gender"]
                # if author has been annotated but book hasnt
        if last_author != name:
            gender = input(f"{name}'s gender: ")
            last_author = name
        try:
            age = input(f"{name}'s age at the time of writing {str(book).encode('latin1').decode('utf8')}: ")
        except UnicodeEncodeError:
            age = input(f"{name}'s age at the time of writing {str(book)}: ")   # sometimes an encoding error pop up

        author_book_items.append({"author_name": name, "author_age": age, "author_gender": gender, "book_name": book})

        # save every time
        wip_author_dict = {"books": author_book_items}
        with open("author_dict.json", "w", encoding="utf-8") as f:
            json.dump(wip_author_dict, f)

    except KeyboardInterrupt:
        break

wip_author_dict = {"books": author_book_items}
with open("author_dict.json", "w", encoding="utf-8") as f:
    json.dump(wip_author_dict, f)

