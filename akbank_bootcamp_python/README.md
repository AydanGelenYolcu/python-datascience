```markdown
# Library Management System

This is a simple **Library Management System** implemented in Python. The system allows users to list books, add new books, and remove existing books from a text file (`books.txt`). The books are stored in the format of title, author, release year, and number of pages.

## Features

- **List Books**: Displays a list of all books available in the `books.txt` file, along with details like the book title, author, release year, and number of pages.
- **Add Book**: Allows the user to add new books to the `books.txt` file.
- **Remove Book**: Enables the user to remove a specific book from the library by entering its title.

## Usage

1. Clone or download the project.
2. Make sure to have Python installed.
3. Create an empty file named `books.txt` in the same directory as the code. This file will store the details of the books in the following format:
    ```
    Book Title, Author Name, Release Year, Number of Pages
    ```

4. Run the script and interact with the menu to manage the books.

## Code Explanation

- `Library.__init__`: Initializes the `Library` class, opens (or creates) the `books.txt` file in append mode, allowing the system to add books.
  
- `Library.__del__`: Closes the file safely when the library instance is deleted.

- `list_books`: Reads the content of the `books.txt` file, splits the lines, and displays each book's details (title, author, release year, and number of pages).

- `add_book`: Prompts the user for book details and adds the new book information to the `books.txt` file.

- `remove_book`: Searches for a book by title in the `books.txt` file and removes it from the library. After removal, the contents are updated in the file.

## Running the Program

Once you run the program, you will be presented with a menu that looks like this:

```
*** MENU ***
1) List Books
2) Add Book
3) Remove Book
q) Exit
```

You can choose any option:

- Enter `1` to list all books.
- Enter `2` to add a new book (you will be asked for book details such as title, author, release year, and number of pages).
- Enter `3` to remove a book by entering its title.
- Enter `q` to exit the program.

### Example Interaction

```
*** MENU ***
1) List Books
2) Add Book
3) Remove Book
q) Exit
Enter your choice (1-3, q): 1

List of Books:
Book: The Alchemist, Author: Paulo Coelho, Release Year: 1988, Pages: 208
Book: Harry Potter, Author: J.K. Rowling, Release Year: 1997, Pages: 223

*** MENU ***
1) List Books
2) Add Book
3) Remove Book
q) Exit
Enter your choice (1-3, q): 2

Enter the book title: 1984
Enter the book author: George Orwell
Enter the release year: 1949
Enter the number of pages: 328
Book '1984' added successfully.
```

## Notes

- The `books.txt` file should be located in the same directory as the Python script. If the file does not exist, the system will create it automatically.
- Make sure that the input for the book information is correct (comma-separated values: `Title, Author, Release Year, Number of Pages`).
  
Enjoy managing your library!
```
