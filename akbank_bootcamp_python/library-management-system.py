class Library:
    def __init__(self):
        self.file = open("books.txt", "a+")
        print("Library initialized. Opening books.txt.")

    def __del__(self):
        #nesne silinince dosyayı kapat
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()
            print("kitaplık kapandı, dosya kapandı")

    def list_books(self):
        # Dosyanın içeriğini oku
        self.file.seek(0)
        content = self.file.read()

        #splitlines() metot
        books_list = content.splitlines()

        # kitap isimleri ve yazarlar
        print("List of Books:")
        for book_info in books_list:
            book_details = book_info.split(', ')
            if len(book_details) == 4:
                book_name, author, release_year, num_pages = book_details
                print(f"Book: {book_name}, Author: {author}, Release Year: {release_year}, Pages: {num_pages}")
            else:
                print(f"Invalid book information: {book_info}")

    def add_book(self):
        # kullanıcı girişi sorgusu
        book_title = input("Enter the book title: ")
        book_author = input("Enter the book author: ")
        release_year = input("Enter the release year: ")
        num_pages = input("Enter the number of pages: ")

        # string
        book_info = f"{book_title}, {book_author}, {release_year}, {num_pages}"

        # ekleme
        self.file.write(book_info + '\n')
        print(f"Book '{book_title}' added successfully.")

    def remove_book(self):
        # kaldırma
        book_title_to_remove = input("Enter the title of the book to remove: ")

        # içerik ve ekleme
        self.file.seek(0)
        content = self.file.read()
        books_list = content.splitlines()

        # listedki silinecek kitapların indeksi
        index_to_remove = -1
        for i, book_info in enumerate(books_list):
            if book_title_to_remove in book_info:
                index_to_remove = i
                break

        if index_to_remove != -1:
            # kitabı kaldır
            removed_book = books_list.pop(index_to_remove)
            print(f"Book '{removed_book}' removed successfully.")

            # içeriği kaldır
            self.file.truncate(0)

        
            for book_info in books_list:
                self.file.write(book_info + '\n')
        else:
            print(f"Book with title '{book_title_to_remove}' not found.")

lib = Library()

# Menu
while True:
    print("\n*** MENU ***")
    print("1) List Books")
    print("2) Add Book")
    print("3) Remove Book")
    print("q) Exit")

    #menü için giriş al
    choice = input("Enter your choice (1-3, q): ")

    # kontrol
    if choice == "1":
        lib.list_books()
    elif choice == "2":
        lib.add_book()
    elif choice == "3":
        lib.remove_book()
    elif choice.lower() == "q":
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please enter a number between 1 and 3 or 'q'.")
