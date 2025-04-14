/// A simple DataFrame implementation for data analysis
/// 
/// This module provides a DataFrame structure that can store and manipulate
/// tabular data with different column types. It supports various operations
/// like filtering, merging, and column operations.
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::path::Path;

// --- Error Handling ---

/// Custom error types for DataFrame operations
#[derive(Debug)]
pub enum Error {
    /// I/O-related errors
    Io(std::io::Error),
    /// CSV parsing errors
    Csv(csv::Error),
    /// When a specified column cannot be found
    ColumnNotFound(String),
    /// When there's a type mismatch in operations
    TypeMismatch(String),
    /// When dimensions don't match for operations
    DimensionMismatch(String),
    /// General invalid operations
    InvalidOperation(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "IO error: {}", err),
            Error::Csv(err) => write!(f, "CSV error: {}", err),
            Error::ColumnNotFound(label) => write!(f, "Column '{}' not found", label),
            Error::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            Error::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            Error::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<csv::Error> for Error {
    fn from(err: csv::Error) -> Self {
        Error::Csv(err)
    }
}

// --- Data Types ---

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    F64(f64),
    I64(i64),
    Bool(bool),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::String(s) => write!(f, "{}", s),
            Value::F64(v) => write!(f, "{}", v),
            Value::I64(v) => write!(f, "{}", v),
            Value::Bool(v) => write!(f, "{}", v),
        }
    }
}

#[derive(Debug, Clone)]
/// Represents a column in a DataFrame
///
/// Stores a vector of values of the same type
pub struct Column {
    /// The data values stored in the column
    pub data: Vec<Value>,
    /// The type of data stored in the column
    data_type: Option<ValueType>,
}

/// Represents the type of values in a column
#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    /// String type
    String,
    /// 64-bit floating point type
    F64,
    /// 64-bit integer type
    I64,
    /// Boolean type
    Bool,
}

impl Column {
    /// Creates a new Column with the specified data type
    fn new(data_type: ValueType) -> Self {
        Column { data: Vec::new(), data_type: Some(data_type) }
    }

    fn add(&mut self, value: Value) -> Result<(), Error> {
         match (&self.data_type, &value) {
            (Some(ValueType::String), Value::String(_)) => (),
            (Some(ValueType::F64), Value::F64(_)) => (),
            (Some(ValueType::I64), Value::I64(_)) => (),
            (Some(ValueType::Bool), Value::Bool(_)) => (),
            (None, _) => { // First element determines the type
                self.data_type = Some(match value {
                    Value::String(_) => ValueType::String,
                    Value::F64(_) => ValueType::F64,
                    Value::I64(_) => ValueType::I64,
                    Value::Bool(_) => ValueType::Bool,
                });
            }
            _ => return Err(Error::TypeMismatch(format!("Attempted to add value {:?} to column of type {:?}", value, self.data_type.as_ref().unwrap()))),
        };
        self.data.push(value);
        Ok(())
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get_type(&self) -> Option<ValueType> {
       self.data_type.clone()
    }
}

/// DataFrame structure to store and manipulate tabular data
///
/// The data is stored in a column-major format, where each column is a vector
/// of values of the same type. Each column has an associated label.
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// Maps column labels to Column instances
    columns: HashMap<String, Column>,
    /// Maintains the original order of columns
    labels: Vec<String>, // Keep track of column order
}

// --- Constructor ---
/// Creates a new DataFrame from labels and data
///
/// # Arguments
/// * `labels` - Vector of column labels
/// * `data` - HashMap mapping column labels to data vectors
///
/// # Returns
/// * `Ok(DataFrame)` if successful
/// * `Err` if there's a dimension mismatch or type inconsistency
pub fn new_dataframe(labels: Vec<String>, data: HashMap<String, Vec<Value>>) -> Result<DataFrame, Error> {
     if labels.is_empty() || data.is_empty() || labels.len() != data.len() {
         return Err(Error::DimensionMismatch("Labels and data must correspond and not be empty".to_string()));
     }

     let mut num_rows: Option<usize> = None;
     let mut columns = HashMap::new();

     for label in &labels {
         let col_data = data.get(label)
             .ok_or_else(|| Error::ColumnNotFound(format!("Data for label '{}' not provided", label)))?;

         if let Some(n_rows) = num_rows {
             if col_data.len() != n_rows {
                 return Err(Error::DimensionMismatch(format!("Column '{}' has {} rows, expected {}", label, col_data.len(), n_rows)));
             }
         } else {
             num_rows = Some(col_data.len());
         }

         if col_data.is_empty() {
             // Handle empty columns - decide on a policy (e.g., error or allow with default type?)
             // For now, let's error if we can't determine type.
             return Err(Error::InvalidOperation(format!("Column '{}' is empty, cannot determine type", label)));
         }

         // Determine column type from the first element
         let first_val_type = match &col_data[0] {
             Value::String(_) => ValueType::String,
             Value::F64(_) => ValueType::F64,
             Value::I64(_) => ValueType::I64,
             Value::Bool(_) => ValueType::Bool,
         };

         let mut column = Column::new(first_val_type.clone());
         for value in col_data {
              // Check type consistency within the provided data
             let current_val_type = match value {
                Value::String(_) => ValueType::String,
                Value::F64(_) => ValueType::F64,
                Value::I64(_) => ValueType::I64,
                Value::Bool(_) => ValueType::Bool,
            };
             if current_val_type != first_val_type {
                 return Err(Error::TypeMismatch(format!("Inconsistent types in column '{}'. Expected {:?}, found {:?}", label, first_val_type, current_val_type)));
             }
             column.add(value.clone())?; // Add clones of the values
         }
         columns.insert(label.clone(), column);
     }

     Ok(DataFrame { columns, labels })
}


// --- DataFrame Methods ---
impl DataFrame {
    /// Returns the number of rows in the DataFrame
    pub fn num_rows(&self) -> usize {
        // Assumes dataframe is valid (all columns have same length)
        self.labels.first().map_or(0, |label| self.columns.get(label).map_or(0, |col| col.len()))
    }

    /// Returns the number of columns in the DataFrame
    pub fn num_cols(&self) -> usize {
        self.labels.len()
    }

    /// Reads data from a CSV file and creates a DataFrame
    /// 
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `col_types` - HashMap specifying the expected type for each column
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` if successful
    /// * `Err` with error details if the operation failed
    pub fn read_csv<P: AsRef<Path>>(
        path: P,
        col_types: &HashMap<String, ValueType> // Expect types: Map Label -> Type
    ) -> Result<DataFrame, Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read header
        let header_line = lines.next().ok_or(Error::InvalidOperation("CSV file is empty".to_string()))??;
        let labels: Vec<String> = header_line.split(',').map(|s| s.trim().to_string()).collect();

        // Validate provided types against header
        if labels.len() != col_types.len() {
            return Err(Error::DimensionMismatch(format!(
                "CSV header has {} columns, but {} types were provided",
                labels.len(),
                col_types.len()
            )));
        }
        for label in &labels {
            if !col_types.contains_key(label) {
                return Err(Error::ColumnNotFound(format!(
                    "Type for column '{}' from CSV header not provided",
                    label
                )));
            }
        }

        let mut data: HashMap<String, Vec<Value>> = HashMap::new();
        for label in &labels {
            data.insert(label.clone(), Vec::new());
        }

        // Read data rows
        for line_result in lines {
            let line = line_result?;
            let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

            if values.len() != labels.len() {
                return Err(Error::DimensionMismatch(format!(
                    "CSV row has {} columns, expected {} columns based on header",
                    values.len(), labels.len()
                )));
            }

            for (i, label) in labels.iter().enumerate() {
                let value_str = values[i];
                let expected_type = col_types.get(label).unwrap(); // We know it exists

                let value = match expected_type {
                    ValueType::String => Value::String(value_str.to_string()),
                    ValueType::F64 => Value::F64(value_str.parse::<f64>().map_err(|e| Error::TypeMismatch(format!("Cannot parse '{}' as f64 for column '{}': {}", value_str, label, e)))?),
                    ValueType::I64 => Value::I64(value_str.parse::<i64>().map_err(|e| Error::TypeMismatch(format!("Cannot parse '{}' as i64 for column '{}': {}", value_str, label, e)))?),
                    ValueType::Bool => Value::Bool(value_str.parse::<bool>().map_err(|e| Error::TypeMismatch(format!("Cannot parse '{}' as bool for column '{}': {}", value_str, label, e)))?),
                };
                data.get_mut(label).unwrap().push(value);
            }
        }

        new_dataframe(labels, data)
    }

    /// Adds a new column to the DataFrame
    /// 
    /// # Arguments
    /// * `label` - Label for the new column
    /// * `data` - Vector of values for the new column
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` with the new DataFrame if successful
    /// * `Err` if the column already exists or lengths don't match
    pub fn add_column(&self, label: String, data: Vec<Value>) -> Result<DataFrame, Error> {
        if self.columns.contains_key(&label) {
            return Err(Error::InvalidOperation(format!("Column '{}' already exists", label)));
        }
        let num_rows = self.num_rows();
        if num_rows != data.len() && !self.columns.is_empty() { // Allow adding first column
             return Err(Error::DimensionMismatch(format!(
                "DataFrame has {} rows, but new column '{}' has {} rows",
                num_rows,
                label,
                data.len()
            )));
        }
        if data.is_empty() {
             return Err(Error::InvalidOperation(format!("Cannot add empty column '{}'", label)));
        }

        // Determine and validate type of new column data
        let first_val_type = match &data[0] {
            Value::String(_) => ValueType::String,
            Value::F64(_) => ValueType::F64,
            Value::I64(_) => ValueType::I64,
            Value::Bool(_) => ValueType::Bool,
        };
        let mut new_col = Column::new(first_val_type.clone());
        for value in data {
             let current_val_type = match &value {
                Value::String(_) => ValueType::String,
                Value::F64(_) => ValueType::F64,
                Value::I64(_) => ValueType::I64,
                Value::Bool(_) => ValueType::Bool,
            };
             if current_val_type != first_val_type {
                 return Err(Error::TypeMismatch(format!("Inconsistent types in new column '{}'. Expected {:?}, found {:?}", label, first_val_type, current_val_type)));
             }
             new_col.add(value)?; // Add owned value
        }

        let mut new_columns = self.columns.clone();
        new_columns.insert(label.clone(), new_col);
        let mut new_labels = self.labels.clone();
        new_labels.push(label);

        Ok(DataFrame { columns: new_columns, labels: new_labels })
    }

    /// Merges two DataFrames
    /// 
    /// # Arguments
    /// * `other` - The DataFrame to merge with
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` with the merged DataFrame if successful
    /// * `Err` if the columns don't match or have different types
    pub fn merge_frame(&self, other: &DataFrame) -> Result<DataFrame, Error> {
        if self.labels.len() != other.labels.len() {
            return Err(Error::DimensionMismatch("DataFrames have different number of columns".to_string()));
        }

        // Check if labels and types match in the same order
        for (label_self, label_other) in self.labels.iter().zip(other.labels.iter()) {
            if label_self != label_other {
                 return Err(Error::DimensionMismatch(format!("Column labels do not match: '{}' vs '{}'", label_self, label_other)));
            }
            let col_self = self.columns.get(label_self).unwrap();
            let col_other = other.columns.get(label_other).unwrap();

            if col_self.get_type() != col_other.get_type() {
                return Err(Error::TypeMismatch(format!("Column '{}' has different types: {:?} vs {:?}", label_self, col_self.get_type(), col_other.get_type())));
            }
        }

        let mut merged_columns = self.columns.clone();

        for label in &self.labels {
            let col_self = merged_columns.get_mut(label).unwrap();
            let col_other = other.columns.get(label).unwrap();
            col_self.data.extend_from_slice(&col_other.data);
        }

        Ok(DataFrame {
            columns: merged_columns,
            labels: self.labels.clone(),
        })
    }

    /// Creates a new DataFrame with only the specified columns
    /// 
    /// # Arguments
    /// * `keep_labels` - Vector of column labels to include
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` with the restricted DataFrame if successful
    /// * `Err` if any column is not found
    pub fn restrict_columns(&self, keep_labels: &[String]) -> Result<DataFrame, Error> {
        let mut new_columns = HashMap::new();
        let mut new_labels = Vec::new();

        for label in keep_labels {
            match self.columns.get(label) {
                Some(col) => {
                    new_columns.insert(label.clone(), col.clone());
                    new_labels.push(label.clone());
                }
                None => return Err(Error::ColumnNotFound(label.clone())),
            }
        }

         if new_labels.is_empty() {
             // Technically valid, but maybe return error or specific empty DataFrame?
             // Let's return an empty DF for now.
             return Ok(DataFrame { columns: HashMap::new(), labels: Vec::new() });
         }

        Ok(DataFrame { columns: new_columns, labels: new_labels })
    }

    /// Filters the DataFrame rows based on a condition
    /// 
    /// # Arguments
    /// * `column_label` - Column label to apply the filter on
    /// * `predicate` - Function that returns true for rows to keep
    /// 
    /// # Returns
    /// * `Ok(DataFrame)` with the filtered DataFrame if successful
    /// * `Err` if the column is not found
    pub fn filter<F>(&self, column_label: &str, predicate: F) -> Result<DataFrame, Error>
    where
        F: Fn(&Value) -> bool,
    {
        let target_col = self.columns.get(column_label)
            .ok_or_else(|| Error::ColumnNotFound(column_label.to_string()))?;

        let _num_rows = self.num_rows();
        let mut keep_indices = Vec::new();
        for (i, value) in target_col.data.iter().enumerate() {
            if predicate(value) {
                keep_indices.push(i);
            }
        }

        let mut new_data: HashMap<String, Vec<Value>> = HashMap::new();
        for label in &self.labels {
             new_data.insert(label.clone(), Vec::with_capacity(keep_indices.len()));
        }

        for row_idx in keep_indices {
            for label in &self.labels {
                let original_col = self.columns.get(label).unwrap();
                new_data.get_mut(label).unwrap().push(original_col.data[row_idx].clone());
            }
        }

        new_dataframe(self.labels.clone(), new_data)
    }

    /// Applies an operation to columns and returns the result
    /// 
    /// # Arguments
    /// * `column_labels` - Column labels to operate on
    /// * `operation` - Function to apply to the columns
    /// 
    /// # Returns
    /// * `Ok(T)` with the operation result if successful
    /// * `Err` if any column is not found
   pub fn column_op<F, T>(&self, column_labels: &[String], operation: F) -> Result<T, Error>
    where
        F: Fn(&[&Column]) -> Result<T, Error>, // Closure takes slice of Column references
    {
        let mut op_columns = Vec::with_capacity(column_labels.len());
        for label in column_labels {
            let col = self.columns.get(label)
                .ok_or_else(|| Error::ColumnNotFound(label.clone()))?;
            op_columns.push(col);
        }

        operation(&op_columns)
    }

    // --- Methods using column_op ---

    /// Calculates the median of a numerical column
    /// 
    /// # Arguments
    /// * `column_label` - Column label to calculate median for
    /// 
    /// # Returns
    /// * `Ok(Value)` with the median if successful
    /// * `Err` if the column is not found or not numerical
    pub fn median(&self, column_label: &str) -> Result<Value, Error> {
        self.column_op(&[column_label.to_string()], |cols| {
            let col = cols[0];
            match col.get_type() {
                Some(ValueType::F64) | Some(ValueType::I64) => {
                    let mut numbers: Vec<f64> = col.data.iter().filter_map(|v| {
                        match v {
                            Value::F64(f) => Some(*f),
                            Value::I64(i) => Some(*i as f64),
                            _ => None, // Should not happen if type check passed
                        }
                    }).collect();

                    if numbers.is_empty() {
                        return Err(Error::InvalidOperation("Cannot compute median of empty column".to_string()));
                    }

                    // Sort numerically
                    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let mid = numbers.len() / 2;
                    let median_val = if numbers.len() % 2 == 0 {
                        (numbers[mid - 1] + numbers[mid]) / 2.0
                    } else {
                        numbers[mid]
                    };
                    // Return as F64, even if original was I64, to handle potential .5
                    Ok(Value::F64(median_val))
                }
                Some(t) => Err(Error::TypeMismatch(format!("Median requires a numeric column (F64 or I64), found {:?}", t))),
                None => Err(Error::InvalidOperation("Cannot compute median of column with unknown type".to_string())),
            }
        })
    }

    /// Performs subtraction between two numerical columns
    /// 
    /// # Arguments
    /// * `label1` - First column label (minuend)
    /// * `label2` - Second column label (subtrahend)
    /// 
    /// # Returns
    /// * `Ok(Vec<Value>)` with the subtraction results if successful
    /// * `Err` if any column is not found or types are incompatible
    pub fn sub_columns(&self, label1: &str, label2: &str) -> Result<Vec<Value>, Error> {
        self.column_op(&[label1.to_string(), label2.to_string()], |cols| {
            let col1 = cols[0];
            let col2 = cols[1];

            if col1.len() != col2.len() {
                // This check might be redundant if column_op ensures columns are valid
                // but good for explicit error handling within the operation itself.
                 return Err(Error::DimensionMismatch("Columns for subtraction have different lengths".to_string()));
            }

            let mut result_vec = Vec::with_capacity(col1.len());

            for (val1, val2) in col1.data.iter().zip(col2.data.iter()) {
                match (val1, val2) {
                    (Value::F64(v1), Value::F64(v2)) => result_vec.push(Value::F64(v1 - v2)),
                    (Value::I64(v1), Value::I64(v2)) => result_vec.push(Value::I64(v1 - v2)),
                    (Value::F64(v1), Value::I64(v2)) => result_vec.push(Value::F64(*v1 - (*v2 as f64))),
                    (Value::I64(v1), Value::F64(v2)) => result_vec.push(Value::F64((*v1 as f64) - *v2)),
                    (_, _) => return Err(Error::TypeMismatch(format!(
                        "Subtraction requires numeric columns (F64 or I64). Found types {:?} and {:?} at corresponding rows.",
                         val1, val2 // Simpler error message as labels aren't directly available here
                    )))
                }
            }
            // The operation closure now returns the Vec<Value>
            Ok(result_vec)
        })
    }
}

// --- Display Implementation ---
impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.labels.is_empty() {
            return writeln!(f, "Empty DataFrame");
        }

        // Determine max width for each column for alignment
        let mut max_widths: HashMap<String, usize> = HashMap::new();
        for label in &self.labels {
            let header_len = label.len();
            let max_data_len = self.columns.get(label)
                .map(|col| col.data.iter().map(|v| v.to_string().len()).max().unwrap_or(0))
                .unwrap_or(0);
            max_widths.insert(label.clone(), std::cmp::max(header_len, max_data_len));
        }

        // Print header
        for (i, label) in self.labels.iter().enumerate() {
            let width = max_widths.get(label).unwrap();
            write!(f, "{:width$}", label, width = width)?; // Left-align header
            if i < self.labels.len() - 1 {
                write!(f, "  ")?;
            }
        }
        writeln!(f)?;

        // Print separator
        for (i, label) in self.labels.iter().enumerate() {
            let width = max_widths.get(label).unwrap();
            write!(f, "{}", "-".repeat(*width))?;
            if i < self.labels.len() - 1 {
                write!(f, "--")?;
            }
        }
        writeln!(f)?;

        // Print rows
        let num_rows = self.num_rows();
        for row_idx in 0..num_rows {
            for (i, label) in self.labels.iter().enumerate() {
                let col = self.columns.get(label).unwrap();
                let value = &col.data[row_idx];
                let width = max_widths.get(label).unwrap();
                // Right-align numeric, left-align others
                 match value {
                    Value::F64(_) | Value::I64(_) => write!(f, "{:>width$}", value.to_string(), width = width)?,
                    _ => write!(f, "{:<width$}", value.to_string(), width = width)?,
                 }

                if i < self.labels.len() - 1 {
                    write!(f, "  ")?;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
} 