/// A simple DataFrame implementation for data analysis
/// 
/// This module provides a DataFrame structure that can store and manipulate
/// tabular data with different column types. It supports various operations
/// like filtering, merging, and column operations.
use std::error::Error;
use std::fmt;
use std::process;
use std::fs::File;
use std::collections::HashMap;

/// Represents a value in a DataFrame column
/// 
/// Each value can be one of four types:
/// - One: String values
/// - Two: Boolean values
/// - Three: 64-bit floating point values
/// - Four: 64-bit integer values
#[derive(Debug, Clone)]
enum ColumnVal {
    One(String),   // Stores string values
    Two(bool),     // Stores boolean values
    Three(f64),    // Stores floating point values
    Four(i64),     // Stores integer values
}

/// DataFrame structure to store and manipulate tabular data
/// 
/// The data is stored in a column-major format, where each column is a vector
/// of values of the same type. Each column has an associated label.
#[derive(Debug, Clone)]
struct DataFrame {
    /// Column labels
    labels: Vec<String>,
    /// Column data stored as vectors, where each vector represents a column
    columns: Vec<Vec<ColumnVal>>,
}

/// Custom error type for DataFrame operations
#[derive(Debug)]
struct MyError(String);

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}
impl Error for MyError {}

impl DataFrame {
    /// Creates a new empty DataFrame
    fn new() -> Self {
        DataFrame {
            labels: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// Reads data from a CSV file and creates a DataFrame
    /// 
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `types` - Vector of type indicators (1=String, 2=bool, 3=f64, 4=i64)
    /// 
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err` with error details if the operation failed
    fn read_csv(&mut self, path: &str, types: &Vec<u32>) -> Result<(), Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(false)
            .flexible(true)
            .from_path(path)?;
        
        let mut first_row = true;
        let mut data: Vec<Vec<ColumnVal>> = vec![vec![]; types.len()];
        
        for result in rdr.records() {
            let r = result?;
            
            if first_row {
                // Store the column labels from the first row
                for elem in r.iter() {
                    self.labels.push(elem.to_string());
                }
                first_row = false;
                continue;
            }
            
            // Parse each element according to its type
            for (i, elem) in r.iter().enumerate() {
                match types[i] {
                    1 => data[i].push(ColumnVal::One(elem.to_string())),
                    2 => data[i].push(ColumnVal::Two(elem.parse::<bool>().unwrap())),
                    3 => data[i].push(ColumnVal::Three(elem.parse::<f64>().unwrap())),
                    4 => data[i].push(ColumnVal::Four(elem.parse::<i64>().unwrap())),
                    _ => return Err(Box::new(MyError("Unknown type".to_string()))),
                }
            }
        }
        
        self.columns = data;
        Ok(())
    }

    /// Prints the DataFrame to the console
    fn print(&self) {
        // Print the labels
        for label in &self.labels {
            print!("{} ", label);
        }
        println!();
        
        // Print the data
        if self.columns.is_empty() {
            return;
        }
        
        // Iterate through rows and columns to print each value
        let row_count = self.columns[0].len();
        for row in 0..row_count {
            for col in 0..self.columns.len() {
                match &self.columns[col][row] {
                    ColumnVal::One(s) => print!("{} ", s),
                    ColumnVal::Two(b) => print!("{} ", b),
                    ColumnVal::Three(f) => print!("{} ", f),
                    ColumnVal::Four(i) => print!("{} ", i),
                }
            }
            println!();
        }
    }

    /// Adds a new column to the DataFrame
    /// 
    /// # Arguments
    /// * `label` - Label for the new column
    /// * `data` - Vector of values for the new column
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` with the new DataFrame if successful
    /// * `Err` if the column length doesn't match the DataFrame row count
    fn add_column(&mut self, label: String, data: Vec<ColumnVal>) -> Result<DataFrame, Box<dyn Error>> {
        // Check if the new column has the same number of rows
        if !self.columns.is_empty() && self.columns[0].len() != data.len() {
            return Err(Box::new(MyError("Column length does not match DataFrame row count".to_string())));
        }
        
        // Create a new DataFrame with the additional column
        let mut new_df = self.clone();
        new_df.labels.push(label);
        new_df.columns.push(data);
        
        Ok(new_df)
    }

    /// Merges two DataFrames
    /// 
    /// # Arguments
    /// * `other` - The DataFrame to merge with
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` with the merged DataFrame if successful
    /// * `Err` if the columns don't match or have different types
    fn merge_frame(&mut self, other: &DataFrame) -> Result<DataFrame, Box<dyn Error>> {
        // Check if the labels match
        if self.labels != other.labels {
            return Err(Box::new(MyError("Column labels do not match".to_string())));
        }
        
        // Check if column types match
        for i in 0..self.columns.len() {
            if !self.columns[i].is_empty() && !other.columns[i].is_empty() {
                let self_type = std::mem::discriminant(&self.columns[i][0]);
                let other_type = std::mem::discriminant(&other.columns[i][0]);
                
                if self_type != other_type {
                    return Err(Box::new(MyError("Column types do not match".to_string())));
                }
            }
        }
        
        // Create a new DataFrame with merged data
        let mut new_df = DataFrame::new();
        new_df.labels = self.labels.clone();
        
        for i in 0..self.columns.len() {
            let mut merged_column = self.columns[i].clone();
            merged_column.extend(other.columns[i].clone());
            new_df.columns.push(merged_column);
        }
        
        Ok(new_df)
    }

    /// Finds the indices of columns by their labels
    /// 
    /// # Arguments
    /// * `labels` - Vector of column labels to find
    /// 
    /// # Returns
    /// * `Ok(Vec<usize>)` with the column indices if successful
    /// * `Err` if any column is not found
    fn find_columns(&self, labels: &[String]) -> Result<Vec<usize>, Box<dyn Error>> {
        let mut indices = Vec::new();
        
        for label in labels {
            let position = self.labels.iter().position(|l| l == label);
            match position {
                Some(idx) => indices.push(idx),
                None => return Err(Box::new(MyError(format!("Column '{}' not found", label)))),
            }
        }
        
        Ok(indices)
    }

    /// Creates a new DataFrame with only the specified columns
    /// 
    /// # Arguments
    /// * `labels` - Vector of column labels to include
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` with the restricted DataFrame if successful
    /// * `Err` if any column is not found
    fn restrict_columns(&self, labels: &[String]) -> Result<DataFrame, Box<dyn Error>> {
        // Find the indices of the requested columns
        let indices = self.find_columns(labels)?;
        
        // Create a new DataFrame with only the selected columns
        let mut new_df = DataFrame::new();
        for &idx in &indices {
            new_df.labels.push(self.labels[idx].clone());
            new_df.columns.push(self.columns[idx].clone());
        }
        
        Ok(new_df)
    }

    /// Filters the DataFrame rows based on a condition
    /// 
    /// # Arguments
    /// * `label` - Column label to apply the filter on
    /// * `operation` - Function that returns true for rows to keep
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` with the filtered DataFrame if successful
    /// * `Err` if the column is not found
    fn filter(
        &self,
        label: &str,
        operation: fn(&ColumnVal) -> bool,
    ) -> Result<Self, Box<dyn Error>> {
        // Find the index of the column to filter on
        let column_idx = match self.labels.iter().position(|l| l == label) {
            Some(idx) => idx,
            None => return Err(Box::new(MyError(format!("Column '{}' not found", label))))
        };
        
        let mut new_df = DataFrame::new();
        new_df.labels = self.labels.clone();
        
        // Initialize empty columns
        for _ in 0..self.columns.len() {
            new_df.columns.push(Vec::new());
        }
        
        // Apply filter and build new dataframe
        for row in 0..self.columns[column_idx].len() {
            if operation(&self.columns[column_idx][row]) {
                // Keep rows where the operation returns true
                for col in 0..self.columns.len() {
                    new_df.columns[col].push(self.columns[col][row].clone());
                }
            }
        }
        
        Ok(new_df)
    }

    /// Applies an operation to columns and returns the result
    /// 
    /// # Arguments
    /// * `labels` - Column labels to operate on
    /// * `op` - Function to apply to the columns
    /// 
    /// # Returns
    /// * `Ok(Vec<ColumnVal>)` with the operation result if successful
    /// * `Err` if any column is not found
    fn column_op(
        &self,
        labels: &[String],
        op: fn(&[Vec<ColumnVal>]) -> Vec<ColumnVal>,
    ) -> Result<Vec<ColumnVal>, Box<dyn Error>> {
        // Find the indices of the columns to operate on
        let indices = self.find_columns(labels)?;
        
        // Extract the requested columns
        let columns: Vec<Vec<ColumnVal>> = indices.iter()
                                             .map(|&idx| self.columns[idx].clone())
                                             .collect();
        
        // Apply the operation
        Ok(op(&columns))
    }
    
    /// Calculates the median of a numerical column
    /// 
    /// # Arguments
    /// * `column_label` - Column label to calculate median for
    /// 
    /// # Returns
    /// * `Ok(f64)` with the median if successful
    /// * `Err` if the column is not found or not numerical
    fn median(&self, column_label: &str) -> Result<f64, Box<dyn Error>> {
        // Median operation closure
        let median_op = |columns: &[Vec<ColumnVal>]| {
            let column = &columns[0];
            let mut values: Vec<f64> = Vec::new();
            
            // Extract numerical values
            for val in column {
                if let ColumnVal::Three(f) = val {
                    values.push(*f);
                }
            }
            
            // Sort values to find median
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let mid = values.len() / 2;
            let median = if values.len() % 2 == 0 {
                // Average of two middle values for even-length arrays
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                // Middle value for odd-length arrays
                values[mid]
            };
            
            vec![ColumnVal::Three(median)]
        };
        
        // Apply the median operation
        let result = self.column_op(&[column_label.to_string()], median_op)?;
        
        // Extract the result
        if let ColumnVal::Three(median) = &result[0] {
            Ok(*median)
        } else {
            Err(Box::new(MyError("Failed to compute median".to_string())))
        }
    }
    
    /// Performs subtraction between two numerical columns
    /// 
    /// # Arguments
    /// * `col1` - First column label (minuend)
    /// * `col2` - Second column label (subtrahend)
    /// 
    /// # Returns
    /// * `Ok(Vec<ColumnVal>)` with the subtraction results if successful
    /// * `Err` if any column is not found
    fn sub_columns(&self, col1: &str, col2: &str) -> Result<Vec<ColumnVal>, Box<dyn Error>> {
        // Subtraction operation closure
        let sub_op = |columns: &[Vec<ColumnVal>]| {
            let col1 = &columns[0];
            let col2 = &columns[1];
            let mut result = Vec::new();
            
            // Perform subtraction for each row
            for i in 0..col1.len() {
                match (&col1[i], &col2[i]) {
                    (ColumnVal::Four(a), ColumnVal::Four(b)) => result.push(ColumnVal::Four(a - b)),
                    (ColumnVal::Three(a), ColumnVal::Three(b)) => result.push(ColumnVal::Three(a - b)),
                    (ColumnVal::Four(a), ColumnVal::Three(b)) => result.push(ColumnVal::Three(*a as f64 - b)),
                    (ColumnVal::Three(a), ColumnVal::Four(b)) => result.push(ColumnVal::Three(a - *b as f64)),
                    _ => {} // Skip non-numerical values
                }
            }
            
            result
        };
        
        // Apply the subtraction operation
        self.column_op(&[col1.to_string(), col2.to_string()], sub_op)
    }
}

/// The main function demonstrates all the DataFrame operations
fn main() {
    // Create a new DataFrame
    let mut df = DataFrame::new();
    
    // Define column types (1=String, 2=bool, 3=f64, 4=i64)
    let types = vec![1, 4, 3, 4, 4, 2];
    
    // Read data from CSV file
    match df.read_csv("basketball.csv", &types) {
        Ok(_) => println!("CSV file loaded successfully"),
        Err(e) => {
            eprintln!("Error reading CSV: {}", e);
            process::exit(1);
        }
    }
    
    println!("Original DataFrame:");
    df.print();
    
    // Add a new column 'HallOfFame'
    let hall_of_fame = vec![
        ColumnVal::Two(true),  // Kareem
        ColumnVal::Two(true),  // Karl
        ColumnVal::Two(false), // LeBron (not yet, but will be)
        ColumnVal::Two(true),  // Kobe
        ColumnVal::Two(true),  // Michael
    ];
    
    let df_with_hof = match df.add_column("HallOfFame".to_string(), hall_of_fame) {
        Ok(new_df) => {
            println!("\nDataFrame with Hall of Fame column:");
            new_df.print();
            new_df
        },
        Err(e) => {
            eprintln!("Error adding column: {}", e);
            process::exit(1);
        }
    };
    
    // Create another DataFrame for merging
    let mut df2 = DataFrame::new();
    match df2.read_csv("more_players.csv", &types) {
        Ok(_) => println!("\nSecond CSV file loaded successfully"),
        Err(e) => {
            eprintln!("Error reading second CSV: {}", e);
            process::exit(1);
        }
    }
    
    // Merge the DataFrames
    let merged_df = match df.merge_frame(&df2) {
        Ok(new_df) => {
            println!("\nMerged DataFrame:");
            new_df.print();
            new_df
        },
        Err(e) => {
            eprintln!("Error merging DataFrames: {}", e);
            process::exit(1);
        }
    };
    
    // Restrict columns to 'Name' and 'TotalPoints'
    let restricted_df = match merged_df.restrict_columns(&["Name".to_string(), "TotalPoints".to_string()]) {
        Ok(new_df) => {
            println!("\nDataFrame with only Name and TotalPoints:");
            new_df.print();
            new_df
        },
        Err(e) => {
            eprintln!("Error restricting columns: {}", e);
            process::exit(1);
        }
    };
    
    // Filter players with PPG > 25.0
    let ppg_filter = |val: &ColumnVal| -> bool {
        if let ColumnVal::Three(ppg) = val {
            *ppg > 25.0
        } else {
            false
        }
    };
    
    let filtered_df = match merged_df.filter("PPG", ppg_filter) {
        Ok(new_df) => {
            println!("\nPlayers with PPG > 25.0:");
            new_df.print();
            new_df
        },
        Err(e) => {
            eprintln!("Error filtering DataFrame: {}", e);
            process::exit(1);
        }
    };
    
    // Calculate median PPG
    match merged_df.median("PPG") {
        Ok(median) => println!("\nMedian PPG: {}", median),
        Err(e) => eprintln!("Error calculating median: {}", e),
    }
    
    // Subtract YearBorn from TotalPoints
    match merged_df.sub_columns("TotalPoints", "YearBorn") {
        Ok(result) => {
            println!("\nTotalPoints - YearBorn:");
            for val in result {
                match val {
                    ColumnVal::Four(i) => print!("{} ", i),
                    ColumnVal::Three(f) => print!("{} ", f),
                    _ => {}
                }
            }
            println!();
        },
        Err(e) => eprintln!("Error subtracting columns: {}", e),
    }
}
