# Rust DataFrame Project

This project implements a basic DataFrame structure in Rust, similar in concept to those found in libraries like Pandas (Python) or Polars (Rust), but simplified for educational purposes.

## Features

*   Stores data in typed columns (String, f64, i64, bool).
*   Supports an arbitrary number of rows and columns.
*   Associates string labels with columns.
*   Provides methods for:
    *   Reading data from CSV files (`read_csv`).
    *   Adding new columns (`add_column`).
    *   Merging two DataFrames (`merge_frame`).
    *   Selecting a subset of columns (`restrict_columns`).
    *   Filtering rows based on a condition (`filter`).
    *   Applying custom operations to columns (`column_op`).
    *   Calculating the median of a numerical column (`median`).
    *   Performing row-wise subtraction between two numerical columns (`sub_columns`).

## Project Structure

*   `Cargo.toml`: Project manifest and dependencies (uses the `csv` crate).
*   `src/lib.rs`: Contains the core implementation of the `DataFrame`, `Column`, `Value`, and related types and methods.
*   `src/main.rs`: Example usage demonstrating the DataFrame functionalities as per the assignment requirements.
*   `data.csv`, `data2.csv`: Sample CSV data files used by `main.rs`.
*   `README.md`: This file.

## Building and Running

1.  **Prerequisites:** Ensure you have Rust and Cargo installed (see [rustup.rs](https://rustup.rs/)).
2.  **Clone/Download:** Obtain the project files.
3.  **Navigate:** Open your terminal in the `dataframe_project` directory.
4.  **Run:** Execute the example program:
    ```bash
    cargo run
    ```
    This will compile the code and run `src/main.rs`, printing the results of the DataFrame operations to the console.

## Generating Documentation

Rust documentation comments (`///`) have been added to the public items in `src/lib.rs`. You can generate HTML documentation by running:

```bash
cargo doc --open
```

This command will build the documentation and open it in your default web browser. 